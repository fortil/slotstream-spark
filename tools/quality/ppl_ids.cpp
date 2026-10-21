// Spark: perplexity of the pinned llama.cpp on token ids given in a file (F9 differential, plan step 96).
//
//   ppl-ids --ids-file IDS.i32 --windows N --out OUT.jsonl [llama.cpp perplexity flags: -m, -c 2048, -b 2048, -cmoe, ...]
//
// llama-perplexity tokenizes a text itself and scores only the second half of every chunk, so it cannot score the engine's
// windows. This helper takes the engine's own token ids (raw little-endian int32, the ids that tools/quality/ppl_wikitext.py
// hashes), runs the first N non-overlapping windows of n_ctx tokens, each from an empty KV cache and without a BOS token, and
// scores tokens 1..n_ctx-1 of every window (row i against ids[i+1]), the engine's and the oracle's rule. The flags are the
// stock ones: the arguments are parsed by llama.cpp's own common_params_parse with the perplexity example, after this tool's
// three options are removed. One JSON line per window is appended to OUT, flushed as it finishes.
#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include <atomic>
#include <chrono>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

// NLL (double) of rows [0, n_rows) against targets, summed over `n_threads` threads in row order inside each thread.
static double nll_rows(llama_context * ctx, const llama_token * targets, int n_rows, int n_vocab, int n_threads) {
    std::vector<double> part(n_threads, 0.0);
    std::atomic<int> next{0};
    auto work = [&](int t) {
        double s = 0.0;
        for (;;) {
            int i = next.fetch_add(1);
            if (i >= n_rows) break;
            const float * lg = llama_get_logits_ith(ctx, i);
            float mx = lg[0];
            for (int v = 1; v < n_vocab; ++v) mx = lg[v] > mx ? lg[v] : mx;
            double sum = 0.0;
            for (int v = 0; v < n_vocab; ++v) sum += std::exp((double) lg[v] - (double) mx);
            s += (std::log(sum) + (double) mx) - (double) lg[targets[i]];
        }
        part[t] = s;
    };
    std::vector<std::thread> th;
    for (int t = 1; t < n_threads; ++t) th.emplace_back(work, t);
    work(0);
    for (auto & x : th) x.join();
    double total = 0.0;
    for (double p : part) total += p;
    return total;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    std::string ids_file, out_file;
    int windows = 0;
    std::vector<char *> rest;
    for (int i = 0; i < argc; ++i) {
        const std::string a = argv[i];
        if (i > 0 && (a == "--ids-file" || a == "--windows" || a == "--out") && i + 1 < argc) {
            const std::string v = argv[++i];
            if (a == "--ids-file") ids_file = v;
            else if (a == "--out") out_file = v;
            else windows = std::stoi(v);
        } else {
            rest.push_back(argv[i]);
        }
    }
    if (ids_file.empty() || out_file.empty() || windows <= 0) {
        fprintf(stderr, "ppl-ids: --ids-file, --windows and --out are required\n");
        return 2;
    }

    common_params params;
    params.n_ctx = 2048;
    params.escape = false;
    common_init();
    if (!common_params_parse((int) rest.size(), rest.data(), params, LLAMA_EXAMPLE_PERPLEXITY)) {
        return 1;
    }
    const int n_ctx = params.n_ctx;
    if (n_ctx <= 1) {
        fprintf(stderr, "ppl-ids: -c must be > 1\n");
        return 2;
    }
    params.n_parallel = 1;
    params.n_batch = std::min(params.n_batch, n_ctx);

    std::vector<llama_token> ids;
    {
        std::ifstream f(ids_file, std::ios::binary | std::ios::ate);
        if (!f) { fprintf(stderr, "ppl-ids: cannot open %s\n", ids_file.c_str()); return 2; }
        const size_t n = (size_t) f.tellg() / sizeof(int32_t);
        f.seekg(0);
        ids.resize(n);
        f.read((char *) ids.data(), n * sizeof(int32_t));
    }
    if ((size_t) windows * n_ctx > ids.size()) {
        fprintf(stderr, "ppl-ids: %zu ids hold %zu windows of %d, %d requested\n", ids.size(), ids.size() / n_ctx, n_ctx, windows);
        return 2;
    }

    llama_backend_init();
    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();
    auto * ctx = llama_init->context();
    if (model == nullptr || ctx == nullptr) {
        fprintf(stderr, "ppl-ids: unable to load the model or create the context\n");
        return 1;
    }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const int n_threads = std::max(1, params.cpuparams.n_threads);
    const int n_batch = params.n_batch;
    fprintf(stderr, "ppl-ids: n_vocab %d, n_ctx %d, n_batch %d, %d windows, %d scoring threads\n", n_vocab, n_ctx, n_batch, windows, n_threads);

    std::vector<double> rows_nll;
    FILE * out = fopen(out_file.c_str(), "a");
    if (!out) { fprintf(stderr, "ppl-ids: cannot append to %s\n", out_file.c_str()); return 2; }
    common_batch batch(ctx);
    double nll_total = 0.0, secs_total = 0.0;
    for (int k = 0; k < windows; ++k) {
        const size_t start = (size_t) k * n_ctx;
        const auto t0 = std::chrono::high_resolution_clock::now();
        llama_memory_clear(llama_get_memory(ctx), true);
        double nll = 0.0;
        int scored = 0;
        // n_ctx tokens in slices of n_batch; every position except the last of the window keeps its logits
        for (int b0 = 0; b0 < n_ctx; b0 += n_batch) {
            const int bs = std::min(n_batch, n_ctx - b0);
            batch.clear();
            for (int j = 0; j < bs; ++j) {
                const int pos = b0 + j;
                batch.add(ids[start + pos], pos, 0, pos < n_ctx - 1);
            }
            if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
                fprintf(stderr, "ppl-ids: window %d: decode failed\n", k);
                return 1;
            }
            const int rows = (b0 + bs == n_ctx) ? bs - 1 : bs;
            // logits_ith indexes the batch: row j of this slice scores ids[start + b0 + j + 1]
            nll += nll_rows(ctx, ids.data() + start + b0 + 1, rows, n_vocab, n_threads);
            scored += rows;
        }
        llama_synchronize(ctx);
        const double secs = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
        fprintf(out, "{\"window\": %d, \"start\": %zu, \"nll\": %.12f, \"tokens\": %d, \"seconds\": %.3f}\n", k, start, nll, scored, secs);
        fflush(out);
        nll_total += nll;
        secs_total += secs;
        fprintf(stderr, "ppl-ids: window %d/%d nll %.2f (%.4f/token) %.1f s\n", k + 1, windows, nll, nll / scored, secs);
    }
    fclose(out);
    fprintf(stderr, "ppl-ids: done, %d windows, %.1f s, %.2f tok/s\n", windows, secs_total, windows * (double) n_ctx / secs_total);
    llama_backend_free();
    return 0;
}

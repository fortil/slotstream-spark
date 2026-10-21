#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cstdint>
#include <memory>
#include <string>

#include "slotio/control_ring.hpp"
#include "slotio/ledger.hpp"
#include "slotio/pool.hpp"
#include "slotio/prefetch.hpp"
#include "slotio/reader.hpp"
#include "slotio/rows.hpp"
#include "slotio/sweep.hpp"

namespace nb = nanobind;
using namespace slotio;

namespace {
constexpr std::int64_t kAbiRecordBytes = 2764800;  // one packed expert record

MemType parse_mem(const std::string& s) {
  if (s == "host_alloc") return MemType::HostAlloc;
  if (s == "vmm_host_numa") return MemType::VmmHostNuma;
  if (s == "anon_thp") return MemType::AnonThp;
  if (s == "host_malloc_for_tests") return MemType::HostMallocForTests;
  throw std::invalid_argument("mem_type must be host_alloc | vmm_host_numa | anon_thp | host_malloc_for_tests");
}

// A table in pinned, device-mapped host memory: exposes __cuda_array_interface__ so that
// torch.as_tensor(obj, device="cuda") aliases it without a copy.
struct DeviceTable {
  std::uint64_t ptr;
  std::size_t len;
  std::string typestr;
  nb::object owner;  // keeps the Pool alive
  nb::dict cai() const {
    nb::dict d;
    d["shape"] = nb::make_tuple(len);
    d["typestr"] = typestr;
    d["data"] = nb::make_tuple(ptr, false);
    d["version"] = 3;
    return d;
  }
};

using Bytes = nb::ndarray<std::uint8_t, nb::c_contig, nb::device::cpu>;
using I64 = nb::ndarray<const std::int64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using I32 = nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

void need_bytes(const Bytes& dst, std::size_t n) {
  if (dst.size() * dst.itemsize() < n) throw std::invalid_argument("dst buffer too small");
}
}  // namespace

NB_MODULE(slotio, m) {
  m.attr("__version__") = "0.1.0";
  m.def("abi_record_bytes", []() { return kAbiRecordBytes; });

  static nb::exception<IoError> io_error(m, "IoError", PyExc_OSError);
  static nb::exception<Exhausted> exhausted(m, "Exhausted", PyExc_RuntimeError);
  static nb::exception<Timeout> timeout(m, "Timeout", PyExc_TimeoutError);

  nb::enum_<Priority>(m, "Priority")
      .value("Demand", Priority::Demand)
      .value("Speculative", Priority::Speculative)
      .value("Sweep", Priority::Sweep)
      .value("Background", Priority::Background);

  nb::class_<File>(m, "File")
      .def_static(
          "open_direct", [](const std::string& path, std::uint32_t align) { return File::open_direct(path, nullptr, align); },
          nb::arg("path"), nb::arg("align") = 4096)
      .def_prop_ro("size", &File::size)
      .def_prop_ro("align", &File::align)
      .def("check_seal", &File::check_seal);

  nb::class_<ReadFault>(m, "ReadFault")
      .def(nb::init<int, int>(), nb::arg("after_jobs"), nb::arg("err") = 5)
      .def_prop_ro("fired", &ReadFault::fired);

  nb::class_<Reader>(m, "Reader")
      .def(
          "__init__",
          [](Reader* self, int lanes, int total_cap, int spec_cap, std::uint64_t scratch_bytes, bool pin_cores) {
            Reader::Config c;
            c.lanes = lanes;
            c.total_cap = total_cap;
            c.spec_cap = spec_cap;
            c.scratch_bytes = scratch_bytes;
            c.pin_cores = pin_cores;
            new (self) Reader(c);
          },
          nb::arg("lanes") = 12, nb::arg("total_cap") = 12, nb::arg("spec_cap") = 12, nb::arg("scratch_bytes") = 2764800,
          nb::arg("pin_cores") = true)
      .def("set_caps", &Reader::set_caps)
      .def("set_fault", &Reader::set_fault, nb::arg("fault").none())
      .def("lane_cpus", [](const Reader& r) { return r.lane_cpus(); })
      .def("stats", [](const Reader& r) {
        ReaderStats s = r.stats();
        nb::dict d;
        d["jobs"] = s.jobs;
        d["bytes"] = s.bytes;
        d["read_ns"] = s.read_ns;
        d["copy_ns"] = s.copy_ns;
        d["max_inflight"] = s.max_inflight;
        d["batches"] = s.batches;
        d["memcpy_hist"] = s.memcpy_hist;
        return d;
      });

  nb::class_<DeviceTable>(m, "DeviceTable")
      .def_prop_ro("__cuda_array_interface__", &DeviceTable::cai)
      .def_prop_ro("ptr", [](const DeviceTable& t) { return t.ptr; })
      .def("__len__", [](const DeviceTable& t) { return t.len; });

  nb::class_<Pool>(m, "Pool")
      .def(
          "__init__",
          [](Pool* self, const std::string& mem_type, std::uint64_t record_bytes, int slots_per_chunk, std::uint64_t chunk_bytes,
             int layers, int experts, int max_slots) {
            PoolConfig c;
            c.record_bytes = record_bytes;
            c.slots_per_chunk = slots_per_chunk;
            c.chunk_bytes = chunk_bytes;
            c.layers = layers;
            c.experts = experts;
            c.max_slots = max_slots;
            new (self) Pool(parse_mem(mem_type), c);
          },
          nb::arg("mem_type"), nb::arg("record_bytes") = 2764800, nb::arg("slots_per_chunk") = 388,
          nb::arg("chunk_bytes") = 1ull << 30, nb::arg("layers") = 49, nb::arg("experts") = 512, nb::arg("max_slots") = 25088)
      .def("resize_slots", &Pool::resize_slots, nb::call_guard<nb::gil_scoped_release>())
      .def_prop_ro("slots", &Pool::slots)
      .def_prop_ro("chunks", &Pool::chunks)
      .def("owned_bytes", &Pool::owned_bytes)
      .def("map_table",
           [](Pool& p) {
             return DeviceTable{p.map_dev(), p.map_len(), "<i4", nb::borrow(nb::find(&p))};
           })
      .def("slot_ptr_table",
           [](Pool& p) {
             return DeviceTable{p.slot_ptr_dev(), static_cast<std::size_t>(p.config().max_slots), "<u8", nb::borrow(nb::find(&p))};
           })
      .def("map_numpy",
           [](Pool& p) {
             std::size_t shape[1] = {p.map_len()};
             return nb::ndarray<nb::numpy, std::int32_t, nb::ndim<1>>(p.map_host(), 1, shape, nb::find(&p));
           })
      .def("slot_view",
           [](Pool& p, int s) {
             if (s < 0 || s >= p.slots()) throw std::out_of_range("slot");
             std::size_t shape[1] = {static_cast<std::size_t>(p.config().record_bytes)};
             return nb::ndarray<nb::numpy, std::uint8_t, nb::ndim<1>>(p.slot_host(s), 1, shape, nb::find(&p));
           })
      .def("slot_dev", &Pool::slot_dev)
      .def("stats", [](const Pool& p) {
        PoolStats s = p.stats();
        nb::dict d;
        d["slots"] = s.slots;
        d["chunks"] = s.chunks;
        d["owned_bytes"] = s.owned_bytes;
        d["free_ns"] = s.free_ns;
        d["alloc_ns"] = s.alloc_ns;
        return d;
      });

  nb::class_<Ensurer>(m, "Ensurer")
      .def(nb::init<Pool&, Reader&, const File&>(), nb::keep_alive<1, 2>(), nb::keep_alive<1, 3>(), nb::keep_alive<1, 4>())
      .def(
          "ensure",
          [](Ensurer& e, int layer, I32 ids, std::uint64_t gen) {
            std::vector<std::int32_t> out;
            EnsureTiming t;
            {
              nb::gil_scoped_release rel;
              out = e.ensure(layer, ids.data(), ids.size(), gen, &t);
            }
            auto* buf = new std::vector<std::int32_t>(std::move(out));
            nb::capsule cap(buf, [](void* p) noexcept { delete static_cast<std::vector<std::int32_t>*>(p); });
            std::size_t shape[1] = {buf->size()};
            nb::dict td;
            td["total_ns"] = t.total_ns;
            td["read_wall_ns"] = t.read_wall_ns;
            td["copy_tail_ns"] = t.copy_tail_ns;
            td["hits"] = t.hits;
            td["misses"] = t.misses;
            td["adopted"] = t.adopted;
            td["bytes"] = t.bytes;
            return nb::make_tuple(nb::ndarray<nb::numpy, std::int32_t, nb::ndim<1>>(buf->data(), 1, shape, cap), td);
          },
          nb::arg("layer"), nb::arg("ids"), nb::arg("gen") = 0)
      .def("set_progress_ptr", [](Ensurer& e, std::uintptr_t addr) { e.set_progress_ptr(reinterpret_cast<const std::uint64_t*>(addr)); })
      .def("resize", &Ensurer::resize, nb::call_guard<nb::gil_scoped_release>())
      .def("pinned", [](Ensurer& e) { return e.pins().count(); });

  nb::class_<Prefetch>(m, "Prefetch")
      .def(
          "__init__",
          [](Prefetch* self, Ensurer& e, Reader& r, const File& f, int slot_cap, int min_evictable) {
            PrefetchConfig c;
            c.slot_cap = slot_cap;
            c.min_evictable = min_evictable;
            new (self) Prefetch(e, r, f, c);
          },
          nb::arg("ensurer"), nb::arg("reader"), nb::arg("file"), nb::arg("slot_cap") = 64, nb::arg("min_evictable") = 256,
          nb::keep_alive<1, 2>(), nb::keep_alive<1, 3>(), nb::keep_alive<1, 4>())
      .def(
          "issue",
          [](Prefetch& p, int layer, I32 ids) {
            nb::gil_scoped_release rel;
            return p.issue(layer, ids.data(), ids.size());
          },
          nb::arg("layer"), nb::arg("ids"))
      .def("expire_layer", &Prefetch::expire_layer, nb::arg("layer"), nb::call_guard<nb::gil_scoped_release>())
      .def("expire_all", &Prefetch::expire_all, nb::call_guard<nb::gil_scoped_release>())
      .def("outstanding", &Prefetch::outstanding)
      .def("stats", [](const Prefetch& p) {
        PrefetchStats s = p.stats();
        nb::dict d;
        d["issued"] = s.issued;
        d["adopted"] = s.adopted;
        d["adopted_ready"] = s.adopted_ready;
        d["adopted_late"] = s.adopted_late;
        d["discarded"] = s.discarded;
        d["failed"] = s.failed;
        d["refused_evictable"] = s.refused_evictable;
        d["refused_cap"] = s.refused_cap;
        d["skipped"] = s.skipped;
        d["wait_ns"] = s.wait_ns;
        return d;
      });

  nb::class_<Staging>(m, "Staging")
      .def(
          "__init__",
          [](Staging* self, const std::string& mem_type, int groups, int records, std::uint64_t record_bytes) {
            new (self) Staging(parse_mem(mem_type), groups, records, record_bytes);
          },
          nb::arg("mem_type"), nb::arg("groups") = 2, nb::arg("records") = 32, nb::arg("record_bytes") = 2764800)
      .def_prop_ro("bytes", &Staging::bytes)
      .def_prop_ro("mapped_bytes", &Staging::mapped_bytes)
      .def_prop_ro("groups", &Staging::groups)
      .def_prop_ro("records", &Staging::records)
      .def("dev", &Staging::dev, nb::arg("buf"), nb::arg("i"))
      .def(
          "host_view",
          [](Staging& st, int buf, int i) {
            std::size_t shape[1] = {static_cast<std::size_t>(st.record_bytes())};
            return nb::ndarray<nb::numpy, std::uint8_t, nb::ndim<1>>(st.host(buf, i), 1, shape, nb::find(&st));
          },
          nb::arg("buf"), nb::arg("i"));

  nb::class_<SweepTicket>(m, "SweepTicket")
      .def_prop_ro("buf", [](const SweepTicket& t) { return t.buf; })
      .def_prop_ro("n", [](const SweepTicket& t) { return t.n; })
      .def_prop_ro("layer", [](const SweepTicket& t) { return t.layer; })
      .def_prop_ro("admitted", [](const SweepTicket& t) { return t.admitted; })
      .def_prop_ro("admit_refused", [](const SweepTicket& t) { return t.admit_refused; });

  nb::class_<Sweep>(m, "Sweep")
      .def(nb::init<Ensurer&, Reader&, const File&, Staging&>(), nb::keep_alive<1, 2>(), nb::keep_alive<1, 3>(),
           nb::keep_alive<1, 4>(), nb::keep_alive<1, 5>())
      .def(
          "read_group",
          [](Sweep& s, int layer, I32 ids, int buf, std::optional<I32> admit) {
            if (admit) return s.read_group(layer, ids.data(), ids.size(), buf, admit->data(), admit->size());
            return s.read_group(layer, ids.data(), ids.size(), buf);
          },
          nb::arg("layer"), nb::arg("experts"), nb::arg("buf"), nb::arg("admit") = nb::none())
      .def("wait", &Sweep::wait, nb::arg("ticket"), nb::arg("timeout_ms"), nb::call_guard<nb::gil_scoped_release>())
      .def("record_dev", &Sweep::record_dev, nb::arg("ticket"), nb::arg("i"))
      .def("set_generation", &Sweep::set_generation)
      .def("resident_ptr_nopin_touch", &Sweep::resident_ptr_nopin_touch, nb::arg("layer"), nb::arg("e"))
      .def("admit_from_staging", &Sweep::admit_from_staging, nb::arg("layer"), nb::arg("e"), nb::arg("src"));

  nb::class_<NgramTicket>(m, "NgramTicket")
      .def_prop_ro("size", &NgramTicket::size)
      .def_prop_ro("misses", &NgramTicket::misses)
      .def_prop_ro("pending", &NgramTicket::pending);

  nb::class_<Sampler>(m, "Sampler")
      .def(
          "__init__",
          [](Sampler* self, int period_ms, std::size_t ring, bool use_nvml) {
            Sampler::Config c;
            c.period_ms = period_ms;
            c.ring = ring;
            c.use_nvml = use_nvml;
            new (self) Sampler(c);
          },
          nb::arg("period_ms") = 20, nb::arg("ring") = 65536, nb::arg("use_nvml") = true)
      .def(
          "register_pool", [](Sampler& s, const std::string& name, const Pool& p) { s.register_pool(name, &p); },
          nb::keep_alive<1, 3>(), nb::arg("name"), nb::arg("pool"))
      .def(
          "register_bytes",
          [](Sampler& s, const std::string& name, std::uint64_t n) {
            s.register_unified(name, [n]() { return n; });
          },
          nb::arg("name"), nb::arg("bytes"))
      .def("unregister", &Sampler::unregister)
      .def("start", &Sampler::start)
      .def("stop", &Sampler::stop, nb::call_guard<nb::gil_scoped_release>())
      .def_prop_ro("nvml_ok", &Sampler::nvml_ok)
      .def_prop_ro("dropped", &Sampler::dropped)
      .def("drain", [](Sampler& s) {
        nb::list out;
        for (auto& x : s.drain()) {
          nb::dict d;
          d["t_ns"] = x.t_ns;
          d["vmrss"] = x.vmrss;
          d["vmhwm"] = x.vmhwm;
          d["avail"] = x.mem_available;
          d["free"] = x.mem_free;
          d["nvml"] = x.nvml;
          nb::dict u;
          for (auto& kv : x.unified) u[kv.first.c_str()] = kv.second;
          d["unified"] = u;
          out.append(d);
        }
        return out;
      });

  nb::class_<Arbiter>(m, "Arbiter")
      .def(
          "__init__",
          [](Arbiter* self, Reader& reader, const std::string& device, const std::string& hwmon, double throttle_c) {
            Arbiter::Config c;
            c.device = device;
            c.hwmon_path = hwmon;
            c.throttle_c = throttle_c;
            Reader* r = &reader;
            new (self) Arbiter(c, [r]() { return r->stats().bytes; });
          },
          nb::keep_alive<1, 2>(), nb::arg("reader"), nb::arg("device") = "nvme0n1", nb::arg("hwmon") = "",
          nb::arg("throttle_c") = 80.0)
      .def("start", [](Arbiter& a, Reader& r) { a.start(&r); }, nb::keep_alive<1, 2>())
      .def("stop", &Arbiter::stop, nb::call_guard<nb::gil_scoped_release>())
      .def("caps", [](const Arbiter& a) {
        auto c = a.caps();
        return nb::make_tuple(c.total, c.spec);
      });

  nb::class_<NgramRows>(m, "NgramRows")
      .def(nb::init<Reader&, const File&, std::uint32_t, std::size_t>(), nb::keep_alive<1, 2>(), nb::keep_alive<1, 3>(),
           nb::arg("reader"), nb::arg("file"), nb::arg("row_bytes"), nb::arg("cache_rows") = 400000)
      .def(
          "read",
          [](NgramRows& r, I64 gids, Bytes dst) {
            need_bytes(dst, gids.size() * r.row_bytes());
            std::int64_t* g = nullptr;
            std::vector<std::int64_t> copy(gids.data(), gids.data() + gids.size());
            g = copy.data();
            std::uint8_t* d = dst.data();
            nb::gil_scoped_release rel;
            r.read(g, copy.size(), d);
          },
          nb::arg("gids"), nb::arg("dst"))
      .def(
          "prefetch",
          [](NgramRows& r, I64 gids) {
            std::vector<std::int64_t> copy(gids.data(), gids.data() + gids.size());
            nb::gil_scoped_release rel;
            return r.prefetch(copy.data(), copy.size());
          },
          nb::arg("gids"))
      .def(
          "wait",
          [](NgramRows& r, NgramTicket& t, Bytes dst) {
            need_bytes(dst, t.size() * r.row_bytes());
            std::uint8_t* d = dst.data();
            nb::gil_scoped_release rel;
            r.wait(t, d);
          },
          nb::arg("ticket"), nb::arg("dst"))
      .def_prop_ro("row_bytes", &NgramRows::row_bytes)
      .def_prop_ro("rows", &NgramRows::rows)
      .def("cache_stats", [](NgramRows& r) { return nb::make_tuple(r.cache().hits(), r.cache().misses(), r.cache().size()); });

  nb::class_<EmbedRows>(m, "EmbedRows")
      .def(
          "__init__",
          [](EmbedRows* self, Reader& reader, const File& trunk, std::uint64_t w_off, std::uint64_t s_off, std::uint64_t b_off,
             std::uint32_t w_row, std::uint32_t s_row, std::uint32_t b_row, std::uint64_t vocab, std::size_t capacity) {
            EmbedLayout l;
            l.w_off = w_off;
            l.s_off = s_off;
            l.b_off = b_off;
            l.w_row = w_row;
            l.s_row = s_row;
            l.b_row = b_row;
            l.vocab = vocab;
            new (self) EmbedRows(reader, trunk, l, capacity);
          },
          nb::keep_alive<1, 2>(), nb::keep_alive<1, 3>(), nb::arg("reader"), nb::arg("trunk"), nb::arg("w_off"),
          nb::arg("s_off"), nb::arg("b_off"), nb::arg("w_row") = 1280, nb::arg("s_row") = 80, nb::arg("b_row") = 80,
          nb::arg("vocab") = 248320, nb::arg("capacity") = 8192)
      .def(
          "read",
          [](EmbedRows& r, I64 ids, Bytes dst) {
            need_bytes(dst, ids.size() * r.row_bytes());
            std::vector<std::int64_t> copy(ids.data(), ids.data() + ids.size());
            std::uint8_t* d = dst.data();
            nb::gil_scoped_release rel;
            r.read(copy.data(), copy.size(), d);
          },
          nb::arg("ids"), nb::arg("dst"))
      .def_prop_ro("row_bytes", &EmbedRows::row_bytes);

  m.def(
      "wait_seq",
      [](std::uintptr_t addr, std::uint64_t target, std::int64_t timeout_ns) {
        nb::gil_scoped_release rel;
        wait_seq(reinterpret_cast<const std::uint64_t*>(addr), target, timeout_ns);
      },
      nb::arg("addr"), nb::arg("target"), nb::arg("timeout_ns"));

  nb::class_<ControlRing>(m, "ControlRing")
      .def(
          "__init__",
          [](ControlRing* self, std::uintptr_t ready, std::uintptr_t done, std::size_t n) {
            new (self) ControlRing(reinterpret_cast<std::uint32_t*>(ready), reinterpret_cast<std::uint32_t*>(done), n);
          },
          nb::arg("ready_addr"), nb::arg("done_addr"), nb::arg("n"))
      .def("publish", &ControlRing::publish, nb::arg("layer"), nb::arg("value") = 1u)
      .def("publish_from", &ControlRing::publish_from, nb::arg("first_layer"))
      .def("ready", &ControlRing::ready, nb::arg("layer"))
      .def("done", &ControlRing::done, nb::arg("layer"))
      .def("armed", &ControlRing::armed)
      .def("clear", &ControlRing::clear)
      .def(
          "wait_done",
          [](const ControlRing& r, std::size_t layer, std::uint32_t value, std::int64_t timeout_ns) {
            nb::gil_scoped_release rel;
            r.wait_done(layer, value, timeout_ns);
          },
          nb::arg("layer"), nb::arg("value"), nb::arg("timeout_ns"))
      .def_prop_ro("size", &ControlRing::size);
}

#include <cstring>

#include "slotio/rows.hpp"

namespace slotio {

EmbedRows::EmbedRows(Reader& reader, const File& trunk, const EmbedLayout& layout, size_t capacity)
    : reader_(reader), file_(trunk), layout_(layout), cache_(layout.row_bytes(), capacity) {}

void EmbedRows::read(const int64_t* ids, size_t n, uint8_t* dst) {
  const uint32_t rb = layout_.row_bytes();
  std::vector<Job> jobs;
  std::vector<size_t> idx;
  for (size_t i = 0; i < n; ++i) {
    if (ids[i] < 0 || static_cast<uint64_t>(ids[i]) >= layout_.vocab) throw std::invalid_argument("token id out of range");
    uint8_t* d = dst + i * rb;
    if (cache_.get(ids[i], d)) continue;
    const uint64_t id = static_cast<uint64_t>(ids[i]);
    jobs.push_back(Job{&file_, layout_.w_off + id * layout_.w_row, layout_.w_row, d, Priority::Demand});
    jobs.push_back(Job{&file_, layout_.s_off + id * layout_.s_row, layout_.s_row, d + layout_.w_row, Priority::Demand});
    jobs.push_back(
        Job{&file_, layout_.b_off + id * layout_.b_row, layout_.b_row, d + layout_.w_row + layout_.s_row, Priority::Demand});
    idx.push_back(i);
  }
  if (jobs.empty()) return;
  BatchResult r = reader_.join(reader_.submit(jobs.data(), jobs.size()));
  if (!r.ok) throw IoError("embedding row read failed", r.err);
  for (size_t i : idx) cache_.put(ids[i], dst + i * rb);
}

}  // namespace slotio

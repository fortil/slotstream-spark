#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

TEST_CASE("388 records fit in one 1 GiB chunk and 389 do not") {
  constexpr long long record = 2'764'800;
  CHECK(388 * record <= (1LL << 30));
  CHECK(389 * record > (1LL << 30));
}

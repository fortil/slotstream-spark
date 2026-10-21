import pytest

pytestmark = pytest.mark.cuda


def test_triton_smoke():
    from triton_smoke import main
    main()

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


def test_display_guard_exists():
    source = (BASE_DIR / 'main.py').read_text(encoding='utf-8')
    assert 'def display_frame' in source
    assert 'if __name__ == "__main__":' in source

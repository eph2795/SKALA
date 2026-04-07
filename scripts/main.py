from src.metrics import calc_masks
from pathlib import Path
import ujson as json

def main():
    final, average = calc_masks()
    print(average)
    Path("result_freeze.json").write_text(json.dumps(average, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

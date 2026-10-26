
from pathlib import Path
import csv

from src.predict_sign import predict

ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = ROOT / "data" / "extra" / "test"
OUTPUT_FILE = ROOT / "data" / "test_results.csv"

EXTENSIONS = {".jpg", ".jpeg", ".png"}


def main():
    results = []

    for folder, expected in [
        ("valid_signs", "valid_sign"),
        ("not_a_sign", "not_a_sign"),
    ]:
        folder_path = TEST_DIR / folder

        if not folder_path.exists():
            print(f"Missing folder: {folder_path}")
            continue

        for image_path in sorted(folder_path.rglob("*")):
            if image_path.suffix.lower() not in EXTENSIONS:
                continue

            result = predict(image_path.read_bytes())
            predicted = (
                result["label"] if result["status"] == "ok"
                else "REJECTED"
            )

            # A valid sign test image should be named with its
            # expected letter, e.g. A_01.jpg, B_02.png.
            if expected == "valid_sign":
                expected_label = image_path.stem.split("_")[0].upper()
                correct = predicted == expected_label
            else:
                expected_label = "not_a_sign"
                correct = predicted == "REJECTED"

            results.append({
                "image": str(image_path.relative_to(ROOT)),
                "expected": expected_label,
                "predicted": predicted,
                "confidence": result.get("confidence", 0.0),
                "correct": correct,
            })

            print(
                f"{image_path.name}: expected={expected_label}, "
                f"predicted={predicted}, "
                f"confidence={result.get('confidence', 0.0):.2%}, "
                f"correct={correct}"
            )

    if not results:
        print("No test images found. Check the test folders.")
        return

    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    accuracy = sum(r["correct"] for r in results) / len(results)
    print(f"\nTotal images: {len(results)}")
    print(f"Overall accuracy: {accuracy:.2%}")
    print(f"Results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

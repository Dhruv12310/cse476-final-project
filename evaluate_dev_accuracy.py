import json
import re
from pathlib import Path

DEV_DB_PATH = Path("cse476_final_project_dev_data.json")
ANS_DB_PATH = Path("dev_answers.json")


def normalize(s_db: str):
    if not isinstance(s_db, str):
        return ""
    s_db = s_db.strip().lower()

    # Remove punctuation
    s_db = re.sub(r"[^a-z0-9\s]", "", s_db)

    # Map synonyms
    synonyms_db = {
        "stay the same": ["same", "no change", "unchanged"],
        "second": ["2nd", "second place"],
        "first": ["1st", "first place"],
        "third": ["3rd", "third place"],
    }

    for canonical_db, syns_db in synonyms_db.items():
        if s_db in syns_db:
            return canonical_db

    return s_db


def extract_number(s_db: str):
    if not isinstance(s_db, str):
        return None
    m_db = re.search(r"-?\d+(\.\d+)?", s_db)
    return m_db.group(0) if m_db else None


def lenient_match(gold_db, pred_db):
    gold_norm_db = normalize(gold_db)
    pred_norm_db = normalize(pred_db)

    # direct normalized match
    if gold_norm_db == pred_norm_db:
        return True

    # number match (very important)
    gnum_db = extract_number(gold_db)
    pnum_db = extract_number(pred_db)
    if gnum_db is not None and pnum_db is not None:
        if gnum_db == pnum_db:
            return True

    # substring tolerance
    if gold_norm_db in pred_norm_db or pred_norm_db in gold_norm_db:
        return True

    return False


def main():
    dev_db = json.load(open(DEV_DB_PATH, "r", encoding="utf-8"))
    ans_db = json.load(open(ANS_DB_PATH, "r", encoding="utf-8"))

    correct_db = 0
    total_db = len(dev_db)
    wrong_examples_db = []

    for i_db, (q_db, a_db) in enumerate(zip(dev_db, ans_db)):
        gold_db = q_db["output"]
        pred_db = a_db["output"]

        if lenient_match(str(gold_db), str(pred_db)):
            correct_db += 1
        else:
            wrong_examples_db.append({
                "index": i_db,
                "question": q_db["input"],
                "gold": gold_db,
                "pred": pred_db
            })

    print(f"\n===== DEV SET ACCURACY =====")
    print(f"Correct: {correct_db} / {total_db}")
    print(f"Accuracy: {correct_db/total_db:.2%}")
    print(f"Wrong: {len(wrong_examples_db)}")

    print("\nShowing 10 random mistakes:")
    for w_db in wrong_examples_db[:10]:
        print("\n---")
        print(f"Index: {w_db['index']}")
        print(f"Q: {w_db['question']}")
        print(f"Gold: {w_db['gold']}")
        print(f"Pred: {w_db['pred']}")


if __name__ == "__main__":
    main()

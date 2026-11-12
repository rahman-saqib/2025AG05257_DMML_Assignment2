"""Stage 2 - Data Validation.
Checks every table, keeps valid rows in data/processed/ and sends invalid rows
(with the reason) to data/errors/. Also writes reports/validation_report.md.
"""
import pandas as pd

INTER_COLS = ["user_id", "item_id", "rating", "timestamp"]


def flag(reasons, mask, text):
    """Add `text` to the reason of every row where mask is True."""
    reasons[mask] = reasons[mask] + text + ";"


def split_valid_invalid(df, reasons, table, checks, summary):
    """Count failures per check, then split into valid rows and error rows."""
    for check in checks:
        count = int(reasons.str.contains(check, regex=False).sum())
        summary.append({"table": table, "check": check, "rows_failed": count})
    bad = reasons != ""
    errors = df[bad].copy()
    errors["error_reason"] = reasons[bad]
    return df[~bad].copy(), errors


def validate_users(users, summary):
    reasons = pd.Series("", index=users.index)
    age = pd.to_numeric(users["age"], errors="coerce")
    flag(reasons, users[["user_id", "age", "country"]].isna().any(axis=1), "missing_value")
    flag(reasons, age.isna() & users["age"].notna(), "wrong_type_age")
    flag(reasons, ~age.between(5, 100) & age.notna(), "invalid_age")
    flag(reasons, users.duplicated("user_id"), "duplicate_user_id")
    checks = ["missing_value", "wrong_type_age", "invalid_age", "duplicate_user_id"]
    return split_valid_invalid(users, reasons, "users", checks, summary)


def validate_items(items, summary):
    reasons = pd.Series("", index=items.index)
    price = pd.to_numeric(items["price"], errors="coerce")
    flag(reasons, items[["item_id", "title", "category", "price"]].isna().any(axis=1), "missing_value")
    flag(reasons, price.isna() & items["price"].notna(), "wrong_type_price")
    flag(reasons, (price <= 0), "invalid_price")
    flag(reasons, items.duplicated("item_id"), "duplicate_item_id")
    checks = ["missing_value", "wrong_type_price", "invalid_price", "duplicate_item_id"]
    return split_valid_invalid(items, reasons, "items", checks, summary)


def validate_interactions(inter, valid_users, valid_items, summary):
    reasons = pd.Series("", index=inter.index)
    num = inter[INTER_COLS].apply(pd.to_numeric, errors="coerce")   # unreadable values become NaN
    now = pd.Timestamp.now().timestamp()

    flag(reasons, inter[INTER_COLS].isna().any(axis=1), "missing_value")
    for col in INTER_COLS:
        flag(reasons, num[col].isna() & inter[col].notna(), "wrong_type_" + col)
    flag(reasons, ~num["rating"].between(1, 5) & num["rating"].notna(), "invalid_rating")
    flag(reasons, ((num["timestamp"] <= 0) | (num["timestamp"] > now)), "invalid_timestamp")
    flag(reasons, num["user_id"].notna() & ~num["user_id"].isin(valid_users["user_id"]), "unknown_user_id")
    flag(reasons, num["item_id"].notna() & ~num["item_id"].isin(valid_items["item_id"]), "unknown_item_id")
    flag(reasons, num.duplicated(["user_id", "item_id"]), "duplicate_interaction")

    checks = (["missing_value"] + ["wrong_type_" + c for c in INTER_COLS] +
              ["invalid_rating", "invalid_timestamp", "unknown_user_id",
               "unknown_item_id", "duplicate_interaction"])
    valid, errors = split_valid_invalid(inter, reasons, "interactions", checks, summary)
    valid[INTER_COLS] = valid[INTER_COLS].apply(pd.to_numeric)
    valid = valid.astype({"user_id": int, "item_id": int, "timestamp": int})
    return valid, errors


def write_report(summary, totals, path):
    lines = ["# Data Validation Report", "",
             f"Generated on: {pd.Timestamp.now():%Y-%m-%d %H:%M}", "",
             "## Rows per table", "",
             "| Table | Total rows | Valid rows | Invalid rows |", "|---|---|---|---|"]
    for t in totals:
        lines.append(f"| {t['table']} | {t['total']} | {t['valid']} | {t['invalid']} |")
    lines += ["", "## Failures per check", "",
              "| Table | Check | Rows failed |", "|---|---|---|"]
    for s in summary:
        lines.append(f"| {s['table']} | {s['check']} | {s['rows_failed']} |")
    lines += ["", "Invalid rows are saved in `data/errors/` with an `error_reason` column."]
    path.write_text("\n".join(lines), encoding="utf-8")


def run(cfg):
    raw, proc, err = cfg["paths"]["raw"], cfg["paths"]["processed"], cfg["paths"]["errors"]
    users = pd.read_csv(raw / "users.csv")
    items = pd.read_csv(raw / "items.csv")
    inter = pd.read_csv(raw / "interactions.csv")

    summary, totals = [], []
    users_ok, users_bad = validate_users(users, summary)
    items_ok, items_bad = validate_items(items, summary)
    inter_ok, inter_bad = validate_interactions(inter, users_ok, items_ok, summary)

    for name, total, ok, bad in [("users", users, users_ok, users_bad),
                                 ("items", items, items_ok, items_bad),
                                 ("interactions", inter, inter_ok, inter_bad)]:
        ok.to_csv(proc / f"{name}_valid.csv", index=False)
        bad.to_csv(err / f"bad_{name}.csv", index=False)
        totals.append({"table": name, "total": len(total), "valid": len(ok), "invalid": len(bad)})
        print(f"{name}: {len(ok)} valid, {len(bad)} invalid")

    pd.DataFrame(summary).to_csv(cfg["paths"]["reports"] / "validation_summary.csv", index=False)
    write_report(summary, totals, cfg["paths"]["reports"] / "validation_report.md")
    return "validation done"

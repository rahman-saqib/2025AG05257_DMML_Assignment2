# Data Validation Report

Generated on: 2026-11-12 18:50

## Rows per table

| Table | Total rows | Valid rows | Invalid rows |
|---|---|---|---|
| users | 1000 | 1000 | 0 |
| items | 500 | 500 | 0 |
| interactions | 48505 | 48498 | 7 |

## Failures per check

| Table | Check | Rows failed |
|---|---|---|
| users | missing_value | 0 |
| users | wrong_type_age | 0 |
| users | invalid_age | 0 |
| users | duplicate_user_id | 0 |
| items | missing_value | 0 |
| items | wrong_type_price | 0 |
| items | invalid_price | 0 |
| items | duplicate_item_id | 0 |
| interactions | missing_value | 1 |
| interactions | wrong_type_user_id | 0 |
| interactions | wrong_type_item_id | 0 |
| interactions | wrong_type_rating | 1 |
| interactions | wrong_type_timestamp | 0 |
| interactions | invalid_rating | 1 |
| interactions | invalid_timestamp | 1 |
| interactions | unknown_user_id | 1 |
| interactions | unknown_item_id | 1 |
| interactions | duplicate_interaction | 1 |

Invalid rows are saved in `data/errors/` with an `error_reason` column.
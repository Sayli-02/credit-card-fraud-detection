# UPI Transactions 2024 Dataset — Schema Notes

## Overview
- **Dataset**: `data/raw/upi_transactions_2024.csv`
- **Total Rows**: 250,000
- **Total Columns**: 17
- **Missing / Null Values**: 0 across all columns
- **Class Label**: `fraud_flag` (Binary: 0 = Legit, 1 = Fraud)
- **Class Balance**: 
  - Legit (`0`): 249,520 (99.808%)
  - Fraud (`1`): 480 (0.192%)
  - Imbalance Ratio: ~520:1

---

## Column Definitions & Types

| Column Name | Data Type | Sample Values | Role / Notes |
| :--- | :--- | :--- | :--- |
| `transaction id` | String / Object | `TXN0000000001` | Unique transaction ID (Dropped from ML features) |
| `timestamp` | Datetime (Str) | `2024-10-08 15:17:28` | Timestamp of transaction |
| `transaction type` | String | `P2P`, `P2M`, `Bill Payment`, `Recharge` | Categorical feature (4 classes) |
| `merchant_category` | String | `Entertainment`, `Grocery`, `Fuel`, `Shopping`, `Food`, `Other`, `Utilities`, `Transport`, `Healthcare`, `Education` | 10 merchant categories |
| `amount (INR)` | Float / Int | `868`, `477`, `1596` | Transaction amount in Indian Rupees (INR) |
| `transaction_status` | String | `SUCCESS`, `FAILED` | Transaction outcome |
| `sender_age_group` | String | `18-25`, `26-35`, `36-45`, `46-55`, `56+` | Sender demographic bucket (`56+` = Senior) |
| `receiver_age_group`| String | `18-25`, `26-35`, `36-45`, `46-55`, `56+` | Receiver demographic bucket |
| `sender_state` | String | `Delhi`, `Uttar Pradesh`, `Karnataka`, `Telangana`, `Maharashtra`, `Gujarat`, `Rajasthan`, `Tamil Nadu`, `West Bengal`, `Andhra Pradesh` | 10 Indian States |
| `sender_bank` | String | `Axis`, `ICICI`, `Yes Bank`, `IndusInd`, `HDFC`, `Kotak`, `SBI`, `PNB` | 8 Indian Banks |
| `receiver_bank` | String | `SBI`, `Axis`, `PNB`, `Yes Bank`, `IndusInd`, `HDFC`, `Kotak`, `ICICI` | 8 Indian Banks |
| `device_type` | String | `Android`, `iOS`, `Web` | Client device |
| `network_type` | String | `4G`, `5G`, `WiFi`, `3G` | Connection type |
| `fraud_flag` | Int (0/1) | `0`, `1` | **Target Label** |
| `hour_of_day` | Int (0-23) | `15`, `0`, `23` | Hour component of transaction |
| `day_of_week` | String | `Monday`, `Tuesday`, ... | Day of the week |
| `is_weekend` | Int (0/1) | `0`, `1` | Weekend indicator |

---

## Key Domain Observations
1. **No Lat/Long Coordinates**: Unlike Sparkov, location is coarse-grained at the state level (`sender_state`).
2. **Bank Pairing**: Sender and receiver bank fields permit interaction features such as `same_bank_flag` (`sender_bank == receiver_bank`).
3. **Senior Citizen Flag**: `sender_age_group == '56+'` allows evaluating age-targeted fraud vulnerabilities.
4. **Time Patterns**: Late-night transactions (e.g. 0 to 5 AM) show higher fraud probability (~0.25% - 0.30% vs daytime 0.15% - 0.19%).

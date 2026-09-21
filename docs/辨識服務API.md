# 辨識服務 API(初版設計)

API 路徑不帶 `recognition-` 前綴(對應的資料表在 `docs/schema.md` 裡才有 `recognition_` 前綴,那是資料庫內部命名習慣,不是 API 合約的一部分)。使用者從頭到尾只上傳/管理**照片**,不會接觸到 embedding 向量本身。

---

## 通用約定

### 版本

所有端點路徑前綴 `/v1`(例如實際呼叫是 `POST /v1/recognize`)。之後有 breaking change 時開新的版本前綴(`/v2`),舊版本可以繼續跑一段時間,不強迫消費端立刻遷移。這份文件底下每個端點標題為了簡潔略過前綴,實際呼叫都要加上。

### 認證

除了上述版本前綴外,所有端點都要求帶 `Authorization: Bearer <key>` header。`<key>` 是部署時由第三方開發者**自己產生**的字串(例如 `openssl rand -hex 32`),存進自己的 `.env`(例如 `VISION_API_KEY`),辨識服務跟開發者自己的 app 都在同一份 `docker-compose.yml` 裡吃這個環境變數。這是 service-to-service 的共用金鑰,不是給終端使用者的帳號系統——服務本身不簽發、不管理金鑰,每個第三方自架的 instance 各自獨立,沒有中央控制端。

驗證失敗(缺 header 或 key 不對)回 `401`,格式見下方「錯誤格式」。

### 錯誤格式

所有非 2xx 回應統一用這個形狀:

```json
{
  "error": {
    "code": "duplicate_name",
    "message": "entity 'led' already exists"
  }
}
```

`code` 是固定字串,可以直接拿來做判斷分支;`message` 是給人看的除錯訊息,文字內容不保證穩定,不要拿它做邏輯判斷。目前定義的 `code`(之後增加端點會再擴充):

| code | 對應狀態碼 | 情境 |
|---|---|---|
| `unauthorized` | 401 | 缺 `Authorization` header 或 key 不對 |
| `not_found` | 404 | 資源 id 從未存在過 |
| `duplicate_name` | 409 | `name` 撞到唯一約束 |
| `unsupported_image_format` | 415 | 上傳的圖片不是 JPEG/PNG |
| `image_too_large` | 413 | 超過檔案大小或像素尺寸上限 |
| `invalid_request` | 400 | 其他輸入格式錯誤(缺必填欄位等) |

### 圖片上傳限制

所有吃圖片的端點(`/recognize`、`/entities/{entity_id}/instances`、`/instances/{id}/samples`)共用同一組限制:

| 限制 | 數值 |
|---|---|
| 格式 | JPEG、PNG(用實際解碼結果判斷,不信任副檔名/宣稱的 MIME type;其他格式含 HEIC 一律拒絕,回 `unsupported_image_format`) |
| 檔案大小 | ≤ 20 MB(超過回 `image_too_large`;在讀取階段依 `Content-Length` 擋,不用整包讀進記憶體才發現太大) |
| 像素尺寸 | 長邊 ≤ 8000px(超過回 `image_too_large`) |

### 分頁

回傳清單的端點(`GET /entities`、`GET /predictions`)共用同一組分頁參數:

| 參數 | 型態 | 說明 |
|---|---|---|
| page | int | 選填,預設 1 |
| limit | int | 選填,預設 20,上限 100 |

回應除了 `result` 之外多帶一個 `total`(符合條件的總筆數),方便前端算頁數。

---

## 辨識

### `POST /recognize`

吃一張圖片,回傳畫面中每個候選框比對到的結果。每次呼叫會自動寫一筆(或多筆)紀錄到 `recognition_predictions`(見 `docs/schema.md`),不需要呼叫端額外做什麼。

**Input**(`multipart/form-data`)

| 欄位 | 型態 | 說明 |
|---|---|---|
| file | image file | 必填,格式/大小限制見上方「圖片上傳限制」,違反回對應的 `image_too_large`/`unsupported_image_format` |

**Output**(`200 OK`, `application/json`)

```json
{
  "score_threshold": 0.6,
  "result": [
    {
      "bbox": [265.2, 75.3, 465.7, 263.5],
      "instance_id": 17,
      "instance_name": "led_front",
      "entity_id": 5,
      "entity_name": "led",
      "score": 0.71,
      "meets_threshold": true
    },
    {
      "bbox": [176.3, 235.7, 206.5, 265.5],
      "instance_id": 9,
      "instance_name": "led_back",
      "entity_id": 5,
      "entity_name": "led",
      "score": 0.42,
      "meets_threshold": false
    }
  ]
}
```

只要特徵池(`recognition_samples`)裡有任何樣本,`result` 就一律回全域最高分對應的身份猜測,不管分數高低——判斷「這個分數算不算數」的責任交給呼叫端,服務本身只給建議值:每個候選框的 `meets_threshold`(`score >= score_threshold`),以及頂層的 `score_threshold`(這次呼叫實際套用的門檻,對應 `recognition_config.score_threshold`)。呼叫端可以直接採用 `meets_threshold`,也可以自己拿 `score` 套更嚴或更鬆的門檻。

`entity_name` 固定回 `"unknown"`、其餘 id 欄位是 `null`,只發生在特徵池完全沒有任何樣本可比對的情況(還沒建過任何 entity/instance,或全部被刪除)。

---

## Entities(概念層)

### `POST /entities`

建立一個新的概念層(不帶樣本圖片,純佔位/建檔用)。

**Input**(`application/json`)

```json
{"name": "led"}
```

**Output**(`201 Created`;`name` 撞到已存在的 entity 名稱回 `409 duplicate_name`,見上方「錯誤格式」)

```json
{"id": 5, "name": "led", "instance_count": 0, "created_at": "2026-08-10T12:00:00Z"}
```

### `GET /entities`

預設不列出已軟刪除的 entity(`is_deleted = true`);要看含已刪除的完整清單,帶 `?include_deleted=true`。支援分頁(`page`/`limit`,見上方「分頁」)。

**Output**(`200 OK`)

```json
{
  "result": [
    {"id": 5, "name": "led", "instance_count": 2},
    {"id": 6, "name": "button_cap", "instance_count": 1}
  ],
  "total": 2
}
```

### `GET /entities/{id}`

**Output**(`200 OK`,`id` 從未存在過才回 `404`——已軟刪除的 entity 一樣回 `200`,見下方)

```json
{"id": 5, "name": "led", "instance_count": 2, "is_deleted": false, "created_at": "2026-08-10T12:00:00Z"}
```

### `DELETE /entities/{id}`

軟刪除(見 `docs/schema.md` 的 `is_deleted` 說明):不會真的移除那一列,只把 `is_deleted` 設成 `true`,id 永久保留、不重用,連帶把底下所有 instances 也標記 `is_deleted = true`。之後 `/recognize` 比對時會自動排除,但 `GET /entities/{id}` 依然查得到(`is_deleted: true`),消費端至少能查到「這個東西曾經存在、後來被移除了」,而不是一個無法解釋的 `404`。

**Output**(`200 OK`,回傳軟刪除後的狀態;`id` 從未存在過才回 `404`)

```json
{"id": 5, "name": "led", "is_deleted": true}
```

---

## Instances(變體)

### `POST /entities/{entity_id}/instances`

在指定概念層底下建立一個變體,可以順便帶樣本圖片(常見情境:第一次建立就直接帶第一批參考圖)。

**Input**(`multipart/form-data`)

| 欄位 | 型態 | 說明 |
|---|---|---|
| name | text | 選填,不填時預設跟 entity 同名 |
| images | image file[] | 選填 |

帶的每張圖片都會先跑 FastSAM 自動抓物件的 bbox(挑面積最大的候選框,避免背景稀釋掉物件本身的特徵,見 `docs/影像辨識.md` 第 6 節),裁切後才算 embedding。回應把每張圖各自建出的 sample 連同偵測到的 bbox 一起列出來,方便呼叫端畫出框給使用者確認——發現框錯了,對個別 sample 打 `PATCH /samples/{id}` 修正(見下方),不用整個 instance 重建。

**Output**(`201 Created`;`entity_id` 對應的 entity 不存在時回 `404`;`name` 撞到同一個 `entity_id` 底下已存在的變體名稱回 `409 duplicate_name`——不同 entity 底下允許同名)

```json
{
  "id": 18,
  "entity_id": 5,
  "name": "led_back",
  "created_at": "2026-08-10T12:00:00Z",
  "samples": [
    {"id": 41, "bbox": [120.0, 80.0, 340.0, 310.0]},
    {"id": 42, "bbox": [95.0, 60.0, 355.0, 300.0]}
  ]
}
```

### `GET /entities/{entity_id}/instances`

**Output**(`200 OK`)

```json
{
  "result": [
    {"id": 17, "name": "led_front", "sample_count": 3},
    {"id": 18, "name": "led_back", "sample_count": 2}
  ]
}
```

### `DELETE /instances/{id}`

用 instance 自己的 id,不用帶 entity_id(個別資源操作一律用自己的全域唯一 id,不用重複帶父層 id)。軟刪除,邏輯跟 `DELETE /entities/{id}` 一致(標記 `is_deleted = true`,`GET` 依然查得到)。`recognition_samples` 不套用軟刪除,底下的 samples 直接物理刪除。

**Output**(`200 OK`,回傳軟刪除後的狀態;`id` 從未存在過才回 `404`)

```json
{"id": 18, "is_deleted": true}
```

---

## Samples(樣本圖片)

### `POST /instances/{instance_id}/samples`

幫指定變體追加一張參考圖,服務端先跑 FastSAM 自動抓物件的 bbox(挑面積最大的候選框)、裁切後才算 embedding 存進特徵池,馬上生效(下一次 `/recognize` 就會用到)。

**Input**(`multipart/form-data`)

| 欄位 | 型態 | 說明 |
|---|---|---|
| image | image file | 必填 |

**Output**(`201 Created`,`instance_id` 不存在時回 `404`)

```json
{"id": 42, "instance_id": 17, "bbox": [120.0, 80.0, 340.0, 310.0], "created_at": "2026-08-10T12:00:00Z"}
```

`bbox` 是這次自動偵測的結果(找不到候選框時是 `null`,代表整張圖直接拿去算 embedding)。發現框錯了,用下面的 `PATCH /samples/{id}` 修正。

### `PATCH /samples/{id}`

修正自動偵測框錯的 bbox。原圖本來就完整保留(不是只存裁切後的結果),服務端拿存好的原圖用新的 `bbox` 重新裁切、重算 embedding,不用重新上傳照片。

**Input**(`application/json`)

```json
{"bbox": [95.0, 60.0, 355.0, 300.0]}
```

**Output**(`200 OK`,查無資料回 `404`)

```json
{"id": 42, "instance_id": 17, "bbox": [95.0, 60.0, 355.0, 300.0]}
```

服務內部會記錄這是第幾次被修正過的框(`predicted_bbox` 保留最初自動偵測的結果不變,`final_bbox` 更新成這次修正的值,見 `docs/schema.md`),不是這個端點的回應要處理的事,呼叫端不用關心。

### `DELETE /samples/{id}`

移除一張品質不好的樣本圖片(例如照片本身拍壞了,不是框的問題——框的問題用 `PATCH` 修正,不用整張刪除重傳)。

**Output**(`204 No Content`,查無資料回 `404`)

---

## Config(執行期可調參數)

對應「硬要求 1」的延伸:不只樣本池能夠不重啟就調整,辨識行為的參數(例如判定門檻)也一樣。存在 `recognition_config`(見 `docs/schema.md`),永遠只有一列。

### `GET /config`

**Output**(`200 OK`)

```json
{
  "score_threshold": 0.3,
  "crop_padding_ratio": 0.1,
  "max_retained_scan_images": 10000
}
```

### `PATCH /config`

部分更新,只送要改的欄位,馬上生效(下一次 `/recognize` 就會套用;`max_retained_scan_images` 調低後,下一輪清除排程會直接套用新上限)。

**Input**(`application/json`)

```json
{"score_threshold": 0.35}
```

**Output**(`200 OK`,回傳更新後的完整設定)

```json
{
  "score_threshold": 0.35,
  "crop_padding_ratio": 0.1,
  "max_retained_scan_images": 10000
}
```

---

## Predictions(預測紀錄 / 標註後台用)

`/recognize` 每次呼叫自動寫入,這幾個端點是給標註後台/消費端用來查詢待審核項目、提交人工修正結果。

### `GET /predictions?status=pending_review`

**Query 參數**

| 參數 | 型態 | 說明 |
|---|---|---|
| status | text | 選填,`pending_review` / `confirmed`,不填回全部 |
| image_path | text | 選填,查單一張圖底下的所有預測 |
| page / limit | int | 選填,分頁,見上方「分頁」 |

**Output**(`200 OK`)

```json
{
  "result": [
    {
      "id": 101,
      "image_path": "scans/2026-08-10-abcd1234.jpg",
      "predicted_bbox": [265.2, 75.3, 465.7, 263.5],
      "predicted_instance_id": 17,
      "predicted_score": 0.71,
      "annotation_status": "pending_review"
    }
  ],
  "total": 1
}
```

### `PATCH /predictions/{id}`

提交人工確認/修正結果。

**Input**(`application/json`)

```json
{"final_instance_id": 17, "final_bbox": [265.2, 75.3, 465.7, 263.5]}
```

`final_instance_id` 傳 `null` 代表「這個候選框其實不屬於任何已知類別」(修正成沒找到/誤判)。

**Output**(`200 OK`,查無資料回 `404`)

```json
{"id": 101, "annotation_status": "confirmed", "final_instance_id": 17, "final_bbox": [265.2, 75.3, 465.7, 263.5]}
```

送出這個 API 後,伺服器內部會依 `docs/schema.md`「運作中優化」小節的邏輯,拿 `final_bbox` 對應的裁切圖重算 embedding、寫進 `recognition_samples`——消費端不用另外呼叫 `/instances/{id}/samples`,`PATCH /predictions/{id}` 已經包含這一步。

---

## 待確認的設計細節

- **消費端主動得知刪除的機制**:軟刪除只解決「查的時候查得到解釋」,沒解決「消費端會不會主動去查」,要嘛消費端自己排程檢查、要嘛之後做 webhook,兩者都還沒設計

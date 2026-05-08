data/ — Veri dosyaları
========================

sample.csv     Küçük örnek (4 satır), hızlı test için.
sms_uci.csv    Tam veri (≈5574 SMS) — aşağıdaki yöntemlerden biriyle oluştur.

--- UCI linki 502 veriyorsa (sık oluyor) ---

1) Otomatik indirme (önerilen, proje kökünden):

  python scripts/download_sms_dataset.py

  Çıktı: data/sms_uci.csv  (UCI ile aynı tür ham/spam verisi, GitHub aynası)

2) Kaggle (hesap gerekir):

  https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

  İndirilen CSV sütunları farklı olabilir; train.py için text + label sütunlarını eşleştir.

--- UCI çalışırsa (resmi) ---

  https://archive.ics.uci.edu/dataset/228/sms+spam+collection

  ZIP → SMSSpamCollection dosyası:

  python scripts/prepare_uci_sms.py "YOL\SMSSpamCollection" -o data/sms_uci.csv

CSV sütunları: text, label  (0 = ham, 1 = spam)

--- Dürüst doğruluk (train / test ayrımı, önerilen) ---

Kaggle spam.csv (v1/v2) örneği:

  python scripts/split_stratified_sms.py --input data/spam.csv --text-col v2 --label-col v1

  python train.py --data data/sms_train.csv --text-col v2 --label-col v1

  python evaluate.py --test-data data/sms_test.csv --text-col v2 --label-col v1

evaluate.py, eğitimde kullandığın tam CSV ile aynı dosyayı ölçüyorsan uyarı verir;
hocaya sunarken tutulmuş test (--test-data) veya metrics.json’daki %20 test metriklerini kullan.

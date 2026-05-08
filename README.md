# SMS Spam Tespiti

İngilizce kısa metinleri (SMS) **ham (0)** ve **spam (1)** olarak ayıran bir ikili metin sınıflandırıcısı.
**TF-IDF + Logistic Regression** ile eğitilir; **Streamlit** arayüzünden canlı tahmin verir.

*Ders teslimi: **Yapay Zekâ ve Bulut Bilişim***

---

## Proje Belgeleri

Kodun yanında dersin istediği iki ek belge de proje kökünde bulunur. Klonladığınızda / ZIP olarak indirdiğinizde doğrudan açıp görebilirsiniz.

| Belge | İçerik |
|---|---|
| [`Proje Önerisi.docx`](<Proje Önerisi.docx>) | Projenin başlangıç önerisi — amaç, kapsam, planlanan yöntem ve veri seti |
| [`Proje Dökümanı.pdf`](<Proje Dökümanı.pdf>) | **Detaylı proje raporu — çalışan uygulamadan alınmış EKRAN GÖRÜNTÜLERİ ile birlikte.** Eğitim çıktıları, metrik tabloları ve Streamlit arayüzünün ekran resimleri bu belgenin içindedir. |

> **Not (değerlendirici için):** Projeyi kurmadan da çalışan halini görmek istiyorsanız doğrudan **`Proje Dökümanı.pdf`** dosyasını açmanız yeterli — uygulamanın gerçek arayüzüne ve eğitim sonuçlarına ait ekran görüntüleri bu PDF'in içindedir.

---

## Hızlı Başlangıç

### Gereksinimler

- **Python 3.11+** — <https://www.python.org/downloads/> (kurulumda *Add python.exe to PATH* kutusunu işaretleyin)
- **Git** — <https://git-scm.com/download/win> (klonlamak için; ZIP indirmek isterseniz gerekmez)

### 1) Projeyi indir

```powershell
git clone https://github.com/cinceli/yapayzekavebulutbilisim.git
cd yapayzekavebulutbilisim
```

veya GitHub'da **Code → Download ZIP**, indirilen klasörü açın, içine girin.

### 2) Kurulum (tek seferlik)

```powershell
.\kur.bat
```

`.venv` sanal ortamını oluşturur ve paketleri yükler. 1–3 dakika sürer.

### 3) Çalıştır

```powershell
.\test.bat      # birim testleri
.\egit.bat      # demo eğitim (modeli kaydeder)
.\arayuz.bat    # Streamlit arayüzü → http://localhost:8501
```

`arayuz.bat` çalıştığında tarayıcı otomatik açılır. Bir SMS yapıştırıp ham/spam tahminini olasılığıyla görebilirsiniz. Durdurmak için terminale dönüp **Ctrl + C**.

---

## Ne Yapıyor?

**UCI SMS Spam Collection** veri setini kullanarak metni vektörleştirip iki alternatif modelden biriyle sınıflandırır:

```
Ham metin
   │ ön işleme (lowercase, URL/noktalama/stopword temizliği)
   ▼
TF-IDF (unigram + bigram)
   │
   ├──► Logistic Regression           ──► 0 (ham) / 1 (spam)
   └──► TruncatedSVD + Random Forest  ──► 0 (ham) / 1 (spam)
```

Veri büyüklüğü (~5.500 örnek) için derin öğrenme yerine **hızlı, yorumlanabilir ve test edilebilir** klasik yaklaşım tercih edildi.

---

## Kullanım

Tüm komutlar proje kökünden çalıştırılır. Sanal ortam zaten aktifse `python` yeterlidir; değilse komutların başına `.\.venv\Scripts\python.exe` ekleyin.

### Eğitim

```powershell
python train.py --demo
python train.py --data data\sms_uci.csv --text-col text --label-col label --model both
```

Çıktı: `models\pipeline_artifacts.joblib` ve `models\metrics.json`.

### Değerlendirme (ayrı CSV ile)

```powershell
python evaluate.py --data data\sample.csv
python evaluate.py --data data\spam.csv --text-col v2 --label-col v1
```

### Stratifiye train/test bölmesi

```powershell
python scripts\split_stratified_sms.py --input data\spam.csv --text-col v2 --label-col v1
python train.py --data data\sms_train.csv --text-col v2 --label-col v1
python evaluate.py --test-data data\sms_test.csv --text-col v2 --label-col v1
```

### Arayüz

```powershell
streamlit run app.py     # Streamlit (önerilen)
python app_gradio.py     # Gradio (alternatif)
```

---

## Veri Seti

**UCI SMS Spam Collection** — 5.572 İngilizce SMS, ham/spam etiketli. Otomatik indirme:

```powershell
python scripts\download_sms_dataset.py
```

Elinizde ZIP varsa: `python scripts\prepare_uci_sms.py "YOL\SMSSpamCollection" -o data\sms_uci.csv`
Detay: [`data/README_VERI.txt`](data/README_VERI.txt).

---

## Proje Yapısı

```
spam_detector/        # Çekirdek paket (preprocessing, training, pipeline)
train.py              # Eğitim giriş noktası
evaluate.py           # Değerlendirme giriş noktası
app.py                # Streamlit arayüzü
app_gradio.py         # Gradio arayüzü
scripts/              # Veri indirme / hazırlama / bölme yardımcıları
data/                 # CSV dosyaları
models/               # Eğitim çıktıları (*.joblib, metrics.json)
tests/                # pytest birim testleri
docs/                 # Ders notları
*.bat                 # Otomasyon kısayolları: kur, egit, test, arayuz
Proje Önerisi.docx    # Başlangıç önerisi (amaç, kapsam, plan)
Proje Dökümanı.pdf    # Detaylı rapor + ekran görüntüleri
```

---

## Kullanılan Teknolojiler

| Katman | Araç | Görevi |
|---|---|---|
| Dil | Python 3.12 | Çekirdek |
| ML | scikit-learn ≥ 1.8 | TF-IDF, LogisticRegression, RandomForest, TruncatedSVD, metrikler |
| Veri | pandas, numpy | CSV okuma, vektör/matris işlemleri |
| Serileştirme | joblib | Eğitilen modeli kaydet/yükle |
| Arayüz | Streamlit (ana), Gradio (alternatif) | Tarayıcıdan canlı tahmin |
| Test | pytest | Birim testleri |
| IDE / AI | Cursor, VS Code | Kod yazımı ve AI ajan etkileşimi |
| Otomasyon | Windows `.bat` | `kur` / `egit` / `test` / `arayuz` kısayolları |

> **RAG / vektör veritabanı kullanılmadı.** Sınıflandırma görevi tek bir metnin içsel örüntülerinden çıkarım yapar; harici belge tabanına ihtiyaç duymadığı için TF-IDF yeterli oldu.

---

## Geliştirme Süreci (Vibecoding ve Prompt Engineering)

Bu projeyi **vibecoding** yöntemiyle, yani bir AI ajanıyla birlikte geliştirdim. Ancak proje yapay zekâ tarafından kendi başına üretilmiş değil; AI'ı bir yardımcı olarak kullandım, **mimari kararları, parametre seçimlerini ve hata çözümlerini büyük ölçüde kendim verdim**. Çalışmamda kullandığım IDE'ler **Cursor** (birincil, AI ajan modu) ve **VS Code** (terminal ve `pytest` çıktıları için) oldu.

### Prompt Mimarisi: Rol – Bağlam – Görev – Format

Prompt'larımı şu dört katmanlı şablonla yapılandırmaya çalıştım:

1. **Rol** — "Sen bir Python geliştirici ve scikit-learn ile çalışan bir veri bilimcisin."
2. **Bağlam** — "`spam_detector/training.py` zaten TF-IDF + Logistic Regression akışı içeriyor; sparse matrisi koru."
3. **Görev** — "Random Forest için ek bir fonksiyon yaz; girdi sparse olduğu için önce TruncatedSVD ile boyut indirgemesi yap."
4. **Format** — "Tek dosyada, type hint'li, kısa docstring'li."

Bu şablon AI'ın **jenerik kod** üretmesini engelliyor; çıktı projenin yapısına oturuyor.

### Kullandığım Prompt Teknikleri

- **Zero-shot** — Görev tek başına yeterince netse, örnek vermeden istek.
  > *"`preprocessing.py` içine, sadece `re` ve `string` kullanarak tek geçişte temizlik yapan `preprocess_text` fonksiyonu ekle."*

- **Few-shot** — Beklenen davranışı örneklerle göster.
  > *"Etiket normalizasyonu yap: `'ham'→0`, `'HAM '→0`, `'spam'→1`, `'Spam!'→1`, sayısal `0/1` aynen kalsın; eşleşmezse `ValueError`."*

- **Chain-of-Thought (CoT)** — Karar gerektiren yerlerde önce düşündür, sonra kod iste.
  > *"Random Forest'ı TF-IDF üzerinde eğitmek istiyorum ama yavaş. Önce şu üç soruyu cevapla, sonra kodu yaz: (1) TF-IDF çıktısı sparse mı dense mi olmalı? (2) RF sparse girdiyle nasıl davranır? (3) Boyut indirgeme için PCA / TruncatedSVD / NMF — hangisi ve neden?"*

### AI'ın Önerilerini Sorguladığım Yerler

- TF-IDF için sabit `min_df=2` önerildi → küçük korpuslarda her kelimeyi eliyordu. **Veri boyutuna göre dinamik `_min_df_for_corpus` fonksiyonunu ben istedim.**
- Random Forest doğrudan TF-IDF üzerinde eğitilecekti → çok yavaştı. **TruncatedSVD ile yoğunlaştırma adımını ben ekletim.**
- Tek model yeterli denildi → proje önerisinde alternatif algoritma istendiği için **iki paralel model kurulması gerektiğini ben hatırlattım.**


## Sonuç ve Özgünlük

Bu çalışmada görev ile ölçek arasındaki dengeyi gözeterek **klasik bir NLP işlem akışı** kurdum. Karmaşık derin öğrenme alternatiflerinden bilinçli olarak kaçındım; problemin doğası, veri büyüklüğü ve yorumlanabilirlik gereksinimi göz önüne alındığında basit modelin doğru kullanımının daha uygun bir tercih olduğunu düşündüm.

AI ajanlarından kapsamlı biçimde yararlandım, ancak şu noktalar büyük ölçüde benim kendi tercihlerimi yansıtıyor:

- **Mimari ayrım:** kütüphane (`spam_detector/`) / giriş noktaları (`train.py`, `app.py`) / test / arayüz sınırlarının çizilmesi
- **Parametre kararları:** `ngram_range=(1,2)`, `sublinear_tf=True`, dinamik `min_df`, `class_weight='balanced'`
- **Performans:** Random Forest için TruncatedSVD ile boyut indirgeme adımının eklenmesi
- **Metrik tercihi:** F1-score'un temel metrik olarak seçilmesi (sadece accuracy yanıltıcıdır)
- **Test stratejisi:** ön işleme ve etiket normalizasyonu için birim test yazılması
- **Kullanılabilirlik:** otomasyon `.bat` dosyaları, hata durumunda pencerenin kapanmaması, `.venv` yokluğunda anlaşılır mesaj

Prompt'larımı **Rol – Bağlam – Görev – Format** şablonu ve **Zero-shot / Few-shot / CoT** teknikleriyle yapılandırdığım için AI çıktıları jenerik kalıplar yerine projenin yapısına yakın biçimde üretildi. Bu süreç bana, vibecoding'in tek başına yeterli olmadığını; **prompt'ların yapılandırılmış biçimde tasarlanmasının da öğrenilmesi gereken ayrı bir pratik** olduğunu öğretti.

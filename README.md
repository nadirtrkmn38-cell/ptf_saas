# PTF Tahmin SaaS Platform ⚡

Türkiye elektrik piyasası (PTF) fiyat tahminleri için Django tabanlı SaaS platformu.

## 🚀 Özellikler

- **72 Saatlik PTF Tahminleri**: XGBoost/LightGBM modelleri ile
- **Kullanıcı Yönetimi**: django-allauth ile kayıt, giriş, email doğrulama
- **Abonelik Sistemi**: Free, Basic, Pro, Enterprise planları
- **Ödeme Entegrasyonu**: iyzico ile Türkiye'de ödeme
- **REST API**: JWT authentication ile güvenli API erişimi
- **Dashboard**: Tailwind + DaisyUI ile modern arayüz
- **Background Tasks**: Celery ile günlük tahmin güncelleme

## 📦 Kurulum

### 1. Repository'yi klonlayın

```bash
git clone https://github.com/your-username/ptf-saas.git
cd ptf-saas
```

### 2. Virtual environment oluşturun

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# veya
venv\Scripts\activate  # Windows
```

### 3. Bağımlılıkları yükleyin

```bash
pip install -r requirements.txt
```

### 4. Environment variables

```bash
cp .env.example .env
# .env dosyasını düzenleyin
```

### 5. Veritabanı

```bash
python manage.py migrate
python manage.py createsuperuser
```

### 6. Geliştirme sunucusu

```bash
python manage.py runserver
```

Tarayıcıda: http://localhost:8000

## 🐳 Docker ile Çalıştırma

```bash
docker-compose up -d
```

Bu komut şunları başlatır:
- Django web uygulaması (port 8000)
- PostgreSQL veritabanı (port 5432)
- Redis cache (port 6379)
- Celery worker
- Celery beat (scheduler)

## 📁 Proje Yapısı

```
ptf_saas/
├── config/                 # Django konfigürasyonu
│   ├── settings/
│   │   ├── base.py        # Temel ayarlar
│   │   ├── development.py # Geliştirme ayarları
│   │   └── production.py  # Production ayarları
│   ├── celery.py          # Celery konfigürasyonu
│   └── urls.py            # Ana URL router
│
├── apps/
│   ├── users/             # Kullanıcı yönetimi
│   ├── subscriptions/     # Abonelik ve ödeme
│   ├── predictions/       # PTF tahminleri
│   ├── api/               # REST API endpoints
│   └── dashboard/         # Web dashboard
│
├── ml_models/             # Makine öğrenmesi kodu
│   ├── data/              # Veri çekme (EPİAŞ, emtia)
│   ├── features/          # Öznitelik mühendisliği
│   └── models/            # Model eğitimi
│
├── templates/             # HTML templates
├── static/                # CSS, JS, images
├── requirements.txt       # Python bağımlılıkları
├── Dockerfile            
├── docker-compose.yml
└── manage.py
```

## 🔌 API Kullanımı

### Authentication (JWT)

```bash
# Token al
curl -X POST http://localhost:8000/api/v1/auth/token/ \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "password"}'

# Response: {"access": "...", "refresh": "..."}
```

### Tahminler

```bash
# 72 saatlik tahminler
curl http://localhost:8000/api/v1/predictions/72h/ \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"

# Günlük özet
curl http://localhost:8000/api/v1/summary/2024-01-15/ \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

## 💳 Abonelik Planları

| Plan | Fiyat | API Limiti | Özellikler |
|------|-------|------------|------------|
| Free | 0₺/ay | 10/gün | Günlük özet |
| Basic | 299₺/ay | 100/gün | Saatlik tahmin |
| Pro | 799₺/ay | 1000/gün | 72 saat, API |
| Enterprise | 2499₺/ay | 10000/gün | SLA, destek |

## ⚙️ Celery Tasks

Zamanlanmış görevler (`config/celery.py`):

- **06:00** - Günlük tahmin güncelleme
- **Her saat** - Cache yenileme
- **00:05** - Abonelik kontrolü
- **00:30** - Gerçek fiyat güncelleme
- **Pazartesi 08:00** - Haftalık performans raporu

## 🔒 Güvenlik

- HTTPS zorunlu (production)
- HSTS aktif
- CSRF koruması
- Rate limiting (plan bazlı)
- JWT token rotation
- Hassas veriler için encryption

## 📊 Model Performansı

Hedef metrikler:
- MAPE: < 15%
- RMSE: < 200 TL
- Doğruluk (<%10 hata): > 70%

## 🛠️ Geliştirme

### Tests

```bash
python manage.py test
```

### Migrations

```bash
python manage.py makemigrations
python manage.py migrate
```

### Static files

```bash
python manage.py collectstatic
```

## 📄 Lisans

MIT License

## 🤝 Katkıda Bulunma

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Commit yapın (`git commit -m 'Add amazing feature'`)
4. Push yapın (`git push origin feature/amazing-feature`)
5. Pull Request açın

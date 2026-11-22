# HAWIYA — Docker Deployment (original AI models)

هذه الملفات **إضافة** إلى فولدر HAWIYA الأصلي وليست مشروعًا جديدًا. لا تستبدلي `engine.py` أو `server.py` أو ملفات JavaScript الحالية.

## ما الذي يظل كما هو؟

- YOLO Iris V2 وArcIris ResNet100 وأوزانهما الحقيقية
- `HAWIYA_AI/hawiya_arciris_cpu_validation.py` وكود ArcIris الأصلي
- معالجة الصور الجديدة من `engine.py` و`server.py`
- صفحة الموقع داخل `static/`

## خطوات التجربة (بعد تثبيت Docker Desktop وتشغيله)

1. انسخي الملفات الأربعة: `Dockerfile`, `.dockerignore`, `compose.yaml`, `requirements.deploy.txt` داخل `C:\Users\VIP\Documents\HAWIYA`.
2. احتفظي بمجلد `HAWIYA_AI` المحلي بنفس الاسم؛ يجب أن يحتوي على الملفين:
   `HAWIYA_AI/models/yolo_iris_v2.pt` و`HAWIYA_AI/models/ResNet100_154000.pt`، إضافة إلى Driver وكود ArcIris.
3. افتحي PowerShell في HAWIYA ونفذي:

   ```powershell
   docker compose build
   docker compose up -d
   docker compose logs -f hawiya
   ```

4. في نافذة PowerShell أخرى:

   ```powershell
   Invoke-RestMethod http://127.0.0.1:8800/api/status
   ```

   **لازم** `ready` تظهر `True`، ثم ارفعي صورة جديدة في الموقع واختبري Analyze وCompare.

5. عند الحاجة لتحضير النتائج التعريفية (لا ينفّذ تلقائيًا):

   ```powershell
   docker compose exec hawiya python PREPARE_SAMPLES.py
   ```

   نتائج المعرض المجهزة مسبقًا ليست بدلًا عن الحساب الفعلي لصور المستخدم، ويجب مراجعة الامتثال لشروط المسابقة.

## GitHub

- ارفعي ملفات Docker الأربعة وملف الشرح فقط بعد المراجعة.
- لا ترفعي `.venv`, `demo_cache`, أو الأوزان `.pt` أو صور عيون خاصة.
- مجلد `HAWIYA_AI` لازم يتوفر على السيرفر بشكل **منفصل** بعد مراجعة تراخيص إعادة التوزيع. هذا الـ Docker Compose يقرأه من مضيف السيرفر ولا يضعه في GitHub أو صورة Docker.
- لو نشرنا باستخدام `git clone` فقط، الموقع **لن ينجح في تحليل الصور** قبل نقل مجلد `HAWIYA_AI` الأصلي إلى السيرفر.

## اعتبارات تشغيل وأمان قبل فتح الموقع للعامة

- لا تعرّضي Port 8800 مباشرة للعالم؛ `compose.yaml` يربطه على localhost فقط. سنضع reverse proxy مع HTTPS قبل فتح الموقع للجنة.
- فعّلي rate limiting وقواعد الحد من الطلبات، وراقبي استهلاك RAM/CPU ومدة المعالجة. الصور الحيوية بيانات حساسة، ويجب توضيح طريقة التعامل معها وتجنّب التخزين غير الضروري.
- السيرفر موجود بكود يجعل 2 jobs بحد أقصى داخل الذاكرة، لكن هذا لا يغني عن تحديد معدل الطلبات عند الـ proxy.
- نظريًا 4GB RAM نقطة بداية؛ لا يوجد إثبات نجاح حتى نبني الصورة ونختبر النموذجين في Linux.
- ملف `engine.py` يقوم بتحميل driver من `HAWIYA_AI`، ولم يتوفر هذا المجلد في النسخة المرفوعة؛ قد توجد تبعيات إضافية تتطلب تعديل Dockerfile بعد الاختبار.
- نتائج `MATCH` و`NO MATCH` ليست أدلة هوية ما لم يكن حد القرار معايرًا باستخدام مجموعة تقييم مستقلة.

## لا تخلطي بين Docker والتقديم

Docker يجعل نشر الـ Backend ممكنًا، لكنه لا يضمن وحده الالتزام بشرط تشغيل **الآلية الأساسية داخل المتصفح** الوارد في تعليمات المسابقة. نحتاج مراجعة مستقلة للتفاعل والحساب الحي قبل التسليم.

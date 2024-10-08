Все видео-модели и модели для изображений похожи друг на друга. Специально была выбрана такая архитектура, чтобы можно было пользоваться результатами обучения старых моделей в новых путём переноса весов в одноимённых слоях. Во всех моделях используется `DDPMScheduler` на `100` шагов зашумления. Все примеры генерации можно увидеть в папке `./results/MovMNIST/`.

История экспериментов:

 0. Примеры кадров из датасета можно увидеть в папке `base_samples`.
 1.  Была обучена модель на изображения. Примеры её генерации лежат в папке `frames`, начальным числом помечено кол-во итераций обучения.
 2. Была обучена модель на изображения с увеличенным батчём. Примеры её генерации лежат в папке `frames_batch96`.
 3. Были обучены модели на видео с батчём размера `2`. Они лежат в папках `uncorr_noise`, `mixed_noise` и `prog_noise`, которые обозначают тип шума, который был использован при обучении.
 4. Была обучена модель на нескоррелированный шум с большим числом итераций. Примеры генерации лежат в папке `uncorr_noise_long_train`. Префикс перед числом обозначает то, какой тип шума был использован на инференсе.
 5. Так как модель показала удовлетворительную чёткость генерации, однако цифры переплывали друг в друга, было решено обучить условную модель. Сначала была обучена условная модель для изображений, примеры генерации можно увидеть в папке `labeled_frames`.
 6. Потом с использованием весов предыдущей модели были обучены условные видео-модели с разными видами шума при обучении. Примеры их генерации можно увидеть в папках `labeled_video`, `labeled_video_mixed_noise` и `labeled_video_prog_noise`. Качество генерации моделей, обученных на любом скоррелированном шуме при любом входном шуме на инференсе оказалось не очень хорошим.
 7. Обучили видео-модель на `mixed_noise(alpha=0.1)` и с эффективным батчём равным реальному(т. е. 2). Стало лучше. Кажется, если подавать такой же шум на инференсе, то цифры меньше плавают. Примеры генерации в `labeled_video_weak_mixed_noise`.
# ImageNet-R Dataset Preparation

Dataset key used by MoTE: `imagenetr`
Setting: B0-Inc20
Classes: 200

Expected ImageFolder paths:

- `/datasets/imagenet-r/train`
- `/datasets/imagenet-r/test`

Validated processed split:

- Train classes: 200
- Test classes: 200
- Train/test class sets equal: yes
- Train images: 24000
- Test images: 6000
- Sample images verified with Pillow: yes
- Download archive integrity checked with `unzip -tq`: yes

Source archive:

- https://drive.google.com/file/d/1SG4TbiL8_DooekztyCVK8mPmfhMo8fkR/view

The downloaded archive itself is intentionally excluded from Git because it is reproducible from the source link and substantially larger than the result artifacts.

# ImageNet-A Dataset Preparation

Dataset key used by MoTE: `imageneta`
Setting: B0-Inc20
Classes: 200

Expected ImageFolder paths:

- `/datasets/imagenet-a/train`
- `/datasets/imagenet-a/test`

Validated processed split:

- Train classes: 200
- Test classes: 200
- Train/test class sets equal: yes
- Train images: 5940
- Test images: 1510
- Sample images verified with Pillow: yes
- Download archive integrity checked with `unzip -tq`: yes

Source archive:

- https://drive.google.com/file/d/19l52ua_vvTtttgVRziCZJjal0TPE9f2p/view

The downloaded archive itself is intentionally excluded from Git because it is reproducible from the source link and substantially larger than the result artifacts.

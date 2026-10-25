## Discord PFP Image DCT Steganography

A Python prototype for hiding short UTF-8 messages in images using DCT coefficient differences and error correction.

See the [detailed technical writeup](https://wqsv.pages.dev/) for details on how it works.

Usage:
```
python stego.py encode carrier.png encoded.png "Hello from Discord."
python stego.py decode encoded.png
```
The maximum message length is **60 bytes**.

### Limitations

Not all images are suitable carriers. Encoding can cause significant pixel clipping, potentially preventing successful decoding.

For example, one test image produced **149,785 clipped pixels**, while another produced 14,946 when encoding the same message. The first image failed to decode, while the second decoded successfully.

These statistics alone cannot predict whether decoding will succeed, so **ALWAYS test the encoded image**. Successfully encoding an image doesn't guarantee that the message can be recovered.

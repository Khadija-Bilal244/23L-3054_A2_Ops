"""Task 4.2 - upload validation. Nothing is written to disk."""
import io

from PIL import Image, UnidentifiedImageError
from werkzeug.utils import secure_filename

ALLOWED_EXT = {"jpg", "jpeg", "png", "webp", "bmp"}
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP", "BMP"}


class ValidationError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.message, self.status = message, status


def read_upload(file_storage, max_bytes):
    """Return (safe_filename, bytes) or raise ValidationError."""
    if file_storage is None or not file_storage.filename:
        raise ValidationError("No file selected. Choose an image and try again.")
    name = secure_filename(file_storage.filename) or "upload"
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if ext not in ALLOWED_EXT:
        raise ValidationError(f"'.{ext or '?'}' files are not supported. Upload a JPG, PNG, WEBP or BMP image.", 415)
    data = file_storage.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValidationError(f"File is larger than {max_bytes // (1024 * 1024)} MB.", 413)
    if not data:
        raise ValidationError("The file is empty.")
    try:  # looks like an image by name - is it really decodable?
        with Image.open(io.BytesIO(data)) as im:
            fmt = im.format
            im.verify()
        with Image.open(io.BytesIO(data)) as im:
            im.load()
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError):
        raise ValidationError("This file has an image extension but could not be decoded as an image.", 422)
    if fmt not in ALLOWED_FORMATS:
        raise ValidationError(f"Image format {fmt} is not supported.", 415)
    return name, data

import logging
from typing import List, Union, Optional, Tuple
import numpy as np
import cv2
from PIL import Image
from pdf2image import convert_from_path, convert_from_bytes
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageProcessor:
    """
    Handles loading, preprocessing, and enhancement of images for OCR.
    """

    @staticmethod
    def load_image(source: Union[str, bytes, Image.Image, np.ndarray]) -> np.ndarray:
        """
        Load an image from various sources (path, bytes, PIL, numpy) into a numpy array (OpenCV format).

        Args:
            source: Image source (file path, bytes, PIL Image, or numpy array).

        Returns:
            np.ndarray: Image in BGR format.

        Raises:
            ValueError: If the input format is unsupported or image cannot be loaded.
        """
        try:
            if isinstance(source, str):
                logger.info(f"Loading image from path: {source}")
                image = cv2.imread(source)
                if image is None:
                    raise ValueError(f"Could not read image from path: {source}")
                return image

            elif isinstance(source, bytes):
                logger.info("Loading image from bytes")
                nparr = np.frombuffer(source, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError("Could not decode image from bytes")
                return image

            elif isinstance(source, Image.Image):
                logger.info("Loading image from PIL Image")
                return cv2.cvtColor(np.array(source), cv2.COLOR_RGB2BGR)

            elif isinstance(source, np.ndarray):
                logger.info("Loading image from numpy array")
                return source

            else:
                raise ValueError(f"Unsupported image source type: {type(source)}")

        except Exception as e:
            logger.error(f"Error loading image: {str(e)}")
            raise e

    @staticmethod
    def convert_pdf_to_images(pdf_source: Union[str, bytes]) -> List[np.ndarray]:
        """
        Convert a PDF file (path or bytes) to a list of images (numpy arrays).

        Args:
            pdf_source: Path to PDF file or bytes content.

        Returns:
            List[np.ndarray]: List of images in BGR format.
        """
        try:
            logger.info("Converting PDF to images...")
            images = []
            
            if isinstance(pdf_source, str):
                pil_images = convert_from_path(pdf_source)
            elif isinstance(pdf_source, bytes):
                pil_images = convert_from_bytes(pdf_source)
            else:
                raise ValueError("PDF source must be a file path (str) or bytes.")

            for pil_img in pil_images:
                # Convert PIL (RGB) to OpenCV (BGR)
                images.append(cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR))
            
            logger.info(f"Converted PDF to {len(images)} images.")
            return images

        except Exception as e:
            logger.error(f"Error converting PDF to images: {str(e)}")
            raise e

    @staticmethod
    def enhance_image(
        image: np.ndarray, 
        deskew: bool = True, 
        denoise: bool = True, 
        contrast: bool = True
    ) -> np.ndarray:
        """
        Apply enhancement techniques to the image for better OCR results.

        Args:
            image: Input image (numpy array).
            deskew: Whether to correct image skew.
            denoise: Whether to apply noise reduction.
            contrast: Whether to apply CLAHE contrast enhancement.

        Returns:
            np.ndarray: Enhanced image.
        """
        try:
            processed = image.copy()

            # Convert to grayscale for processing
            if len(processed.shape) == 3:
                gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
            else:
                gray = processed

            if deskew:
                logger.info("Applying deskewing...")
                # Calculate skew angle
                coords = np.column_stack(np.where(gray > 0))
                angle = cv2.minAreaRect(coords)[-1]
                
                # Correct angle format
                if angle < -45:
                    angle = -(90 + angle)
                else:
                    angle = -angle
                
                # Rotate only if angle is significant
                if abs(angle) > 0.5:
                    (h, w) = processed.shape[:2]
                    center = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(center, angle, 1.0)
                    processed = cv2.warpAffine(
                        processed, M, (w, h), 
                        flags=cv2.INTER_CUBIC, 
                        borderMode=cv2.BORDER_REPLICATE
                    )

            if denoise:
                logger.info("Applying denoising...")
                processed = cv2.fastNlMeansDenoisingColored(processed, None, 10, 10, 7, 21)

            if contrast:
                logger.info("Applying contrast enhancement...")
                # Convert to LAB color space
                lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                
                # Apply CLAHE to L-channel
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                cl = clahe.apply(l)
                
                # Merge channels
                limg = cv2.merge((cl, a, b))
                processed = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

            return processed

        except Exception as e:
            logger.error(f"Error enhancing image: {str(e)}")
            raise e

    @staticmethod
    def resize_for_ocr(image: np.ndarray, target_height: int = 1000) -> np.ndarray:
        """
        Resize image to a target height while maintaining aspect ratio, optimal for OCR.

        Args:
            image: Input image.
            target_height: Desired height in pixels.

        Returns:
            np.ndarray: Resized image.
        """
        try:
            (h, w) = image.shape[:2]
            
            # Use floating point for aspect ratio calculation!
            aspect_ratio = float(w) / float(h)
            new_width = int(target_height * aspect_ratio)
            
            logger.info(f"Resizing image from {w}x{h} to {new_width}x{target_height}")
            
            resized = cv2.resize(image, (new_width, target_height), interpolation=cv2.INTER_AREA)
            return resized

        except Exception as e:
            logger.error(f"Error resizing image: {str(e)}")
            raise e

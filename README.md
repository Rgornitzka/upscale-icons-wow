# WoW Icon Upscaler

A Python script that uses Real-ESRGAN to upscale World of Warcraft achievement icons and other anime-style images with high quality results.

## Features

- **4x Upscaling**: Uses Real-ESRGAN_x4plus_anime_6B model for optimal anime-style image upscaling
- **Multi-processing**: Processes multiple images in parallel for faster results
- **Customizable**: Adjustable cropping, borders, and processing parameters
- **Batch Processing**: Automatically processes all images in the input folder
- **Quality Preservation**: Maintains image quality while significantly increasing resolution

## Prerequisites

- **Python 3.10** (required for compatibility with Real-ESRGAN dependencies)
- **Windows 10/11** (tested on Windows)
- **Git** (for cloning the repository)

## Installation

### Step 1: Install Python 3.10

1. Download Python 3.10 from [python.org](https://www.python.org/downloads/release/python-3100/)
2. Run the installer and **check "Add Python 3.10 to PATH"**
3. Verify installation: `py -3.10 --version`

### Step 2: Clone and Setup Project

```powershell
# Clone the repository
git clone <repository-url>
cd upscale-icons-wow

# Create virtual environment
py -3.10 -m virtualenv venv310

# Activate virtual environment
.\venv310\Scripts\Activate.ps1
```

### Step 3: Install Dependencies

```powershell
# Install all required packages
py -3.10 -m pip install torch torchvision
py -3.10 -m pip install opencv-python
py -3.10 -m pip install realesrgan
```

## Usage

### Basic Usage

1. **Place your images** in the `input/` folder
2. **Run the script:**
   ```powershell
   py -3.10 main.py
   ```
3. **Find upscaled images** in the `output/` folder

### Customization

You can modify the parameters in `main.py`:

```python
upscale_anime_images(
    input_folder="input",           # Input folder path
    output_folder="output",         # Output folder path
    crop_amount=5,                  # Pixels to crop from edges
    border_width=5,                 # Border thickness
    border_color=(0, 0, 0),        # Border color (RGB)
    num_processes=os.cpu_count(),   # Number of parallel processes
)
```

### Supported Image Formats

- PNG (recommended for transparency)
- JPG/JPEG
- WebP

## Project Structure

```
upscale-icons-wow/
├── input/              # Place your images here
├── output/             # Upscaled images appear here
├── weights/            # Model weights (auto-downloaded)
├── models/             # Additional model files
├── main.py             # Main processing script
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Performance

- **Processing Speed**: ~8-10 seconds per image (varies by hardware)
- **Memory Usage**: ~2-4GB RAM recommended
- **GPU Support**: Automatically uses CUDA if available
- **Multi-core**: Utilizes all CPU cores for parallel processing

## Troubleshooting

### Common Issues

1. **"Module not found" errors**

   - Ensure you're using Python 3.10: `py -3.10 --version`
   - Reinstall dependencies: `py -3.10 -m pip install -r requirements.txt`

2. **Slow processing**

   - Reduce `num_processes` parameter
   - Close other applications to free memory
   - Consider using GPU if available

3. **Out of memory errors**
   - Reduce `num_processes` to 1-2
   - Process fewer images at once
   - Close other applications

### Dependencies

- **PyTorch**: Deep learning framework
- **Real-ESRGAN**: Image upscaling library
- **OpenCV**: Image processing
- **BasicSR**: Super-resolution framework

## Model Information

The script uses the **Real-ESRGAN_x4plus_anime_6B** model, which is:

- Optimized for anime-style images
- Provides 4x upscaling
- Automatically downloaded on first run
- ~1.4GB in size

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Feel free to submit issues and enhancement requests!

## Acknowledgments

- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) by xinntao
- [BasicSR](https://github.com/xinntao/BasicSR) framework
- World of Warcraft achievement icons for testing

# 🏥 Medical Viewer with AI

AI-enhanced medical imaging viewer with Multi-Planar Reconstruction (MPR), automatic organ segmentation, and anatomy detection.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## ✨ Key Features

- **Multi-Planar Views**: Axial, Coronal, Sagittal, and Oblique slicing
- **DICOM & NIfTI Support**: Load single files, series, or create MPR from single slice
- **AI Anatomy Detection**: Automatic orientation and region identification (Gemini AI)
- **Auto Segmentation**: 100+ organs with TotalSegmentator
- **3D ROI Tools**: Draw, adjust, and export regions of interest
- **Cross-View Sync**: Click anywhere to update all views
- **Cine Mode**: Animate through slices
- **Export**: Save ROI slices (PNG) or volumes (NIfTI)

## 🚀 Quick Start

```bash
# Install dependencies
pip install SimpleITK numpy pydicom nibabel PyQt5 matplotlib Pillow imageio

# Optional: AI features
pip install google-generativeai TotalSegmentator torch

# Run
python MPR_final_with_api.py
```

## 📖 Basic Usage

### Load Images
- **NIfTI**: Click "Load Main Scan (.nii)" → Select `.nii` or `.nii.gz` file
- **DICOM Series**: Click "Load DICOM Series" → Select folder
- **Single DICOM MPR**: Click "📄 MPR View Single DICOM" → Select `.dcm` file

### AI Analysis
1. Get free API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Click "🔑 Set Gemini API Key" and paste it
3. AI runs automatically on new scans (detects orientation, view, anatomy)
4. For segmentation: Click "🤖 Auto-Segment" (takes 2-5 min)

### Work with ROIs
1. Click "Axial", "Coronal", or "Sagittal" under "Draw ROI on"
2. Drag rectangle on the view
3. Adjust with "🔧 Adjust ROI Range"
4. Export: "📁 Export Slices" (PNG) or "💾 Export ROI to .nii" (volume)

### Navigation
- **Sliders**: Scroll through slices
- **Mouse**: Click any view to jump to that position
- **Cine**: Click "▶ Z/Y/X" to animate
- **Windowing**: Adjust L (level) and W (width) sliders

## 🛠️ Common Issues

| Problem | Solution |
|---------|----------|
| "Gemini API error" | Check API key and internet connection |
| "TotalSegmentator not found" | `pip install TotalSegmentator torch` |
| Slow segmentation | Use GPU: `pip install torch --index-url https://download.pytorch.org/whl/cu118` |
| Non-orthonormal header | Automatically fixed - no action needed |

## 📝 License & Disclaimer

**License**: MIT License

**⚠️ Medical Disclaimer**: This software is for research and educational purposes only. NOT for clinical diagnosis or treatment. Always consult qualified healthcare professionals.

## 🙏 Credits

Built with: SimpleITK • PyQt5 • Matplotlib • Google Gemini • TotalSegmentator

---

**Questions?** Open an [issue](https://github.com/yourusername/medical-viewer/issues) or contact: your.email@example.com

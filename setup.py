from setuptools import setup

setup(
    name="radarbox",
    version="0.1.0",
    description="RadarBox camera-radar fusion boxing system",
    package_dir={"": "src"},
    py_modules=[
        "core.punch_vision_common",
        "core.radar_agent",
        "core.vision_agent",
        "core.fusion_core",
    ], 
    python_requires=">=3.10",
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "scikit-learn",
        "joblib",
        "mediapipe",
        "opencv-contrib-python",
    ],
)
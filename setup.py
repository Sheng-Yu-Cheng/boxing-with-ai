from setuptools import setup

setup(
    name="radarbox",
    version="0.1.0",
    description="RadarBox camera-radar fusion boxing system",
    package_dir={"": "src"},
    py_modules=[
        "punch_vision_common",
        "radar_agent",
        "vision_agent_trajectory",
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
# EnergyBench++ (Under Development)

## Overview

EnergyBench++ is a novel benchmarking framework designed to evaluate the energy efficiency of lightweight deep learning models under edge-like constraints. Our benchmark introduces the **Dynamic Energy-Accuracy Efficiency Ratio (D-EAER)**, which adapts to runtime constraints such as batch size and power limits. This work aims to bridge the gap between energy efficiency and real-world deployment by providing a reproducible, open-source toolkit.

## Key Features

- **Benchmarking on CIFAR-10 and STL-10** using five lightweight models:
  - MobileNetV2
  - ResNet-18
  - EfficientNet-B0
  - SqueezeNet
  - ShuffleNet
- **Evaluation under edge-like constraints**, executed on a GTX 1650 GPU.
- **Dynamic Energy-Accuracy Efficiency Ratio (D-EAER)** for runtime adaptability.
- **Framework for recommending models** for specific edge computing scenarios, including low-power IoT devices.
- **Fully open-source** for accessibility and reproducibility.

## Installation

To set up and run EnergyBench++, follow these steps:

```bash
# Clone the repository
git clone https://github.com/yourusername/EnergyBench++
cd EnergyBench++

# Install dependencies
pip install -r requirements.txt
```

## Usage (To Be Done)

Run the benchmarking framework with:

```bash
python benchmark.py --model resnet18 --dataset cifar10 --power_limit 50
```

To compare multiple models under different constraints:

```bash
python compare_models.py --models mobilenetv2 resnet18 squeezenet --dataset stl10 --batch_size 32
```

## Results & Insights

EnergyBench++ provides insights into energy efficiency trade-offs, including:

- Model performance under **various power constraints**.
- **Optimal model selection** for edge devices.
- **D-EAER scores** for better energy-aware decision-making.

## Contributing

We welcome contributions to improve EnergyBench++! Feel free to:

- Open an issue for feature requests or bug reports.
- Submit a pull request with improvements.

## Citation

If you use EnergyBench++ in your research, please cite our work:

```bibtex
@article{yourpaper2025,
  title={EnergyBench++: A Benchmark for Energy-Efficient Deep Learning on Edge Devices},
  author={Your Name and Others},
  journal={To be published},
  year={2025}
}
```

## License

This project is licensed - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or collaboration inquiries, please reach out to:
[akuma102@uic.edu]

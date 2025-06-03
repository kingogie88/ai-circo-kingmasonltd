# Circo AI: Advanced Plastic Recycling System

## Overview
Circo AI is a cutting-edge, modular AI-powered plastic recycling system capable of processing 1 million tons of plastic annually through distributed facilities. The system combines YOLOv11 computer vision, predictive maintenance, energy optimization, and blockchain logging for complete transparency and efficiency.

## Key Features
- AI Vision System with YOLOv11 for plastic classification
- Robotic Control Interface with ROS2 integration
- Predictive Maintenance Engine
- Energy Management System
- Blockchain Logging Module
- Central Orchestration Hub
- Real-time Analytics Dashboard

## System Requirements

### Hardware Requirements
- Industrial IP cameras (GigE Vision, USB3 Vision, GenICam protocols)
- Robot Arms (Compatible with ABB, KUKA, Fanuc, Universal Robots, Doosan)
- NVIDIA Jetson AGX Orin or Intel NUC with discrete GPU
- Industrial-grade conveyor systems with Modbus TCP/RTU support

### Software Requirements
- Python 3.11+
- CUDA 12.0+
- ROS2 (Robot Operating System)
- Docker & Kubernetes
- PostgreSQL 15+
- Redis 7+
- Node.js 20+

## Installation

1. Clone the repository:
```bash
git clone https://github.com/kingogie88/ai-circo-kingmasonltd.git
cd ai-circo-kingmasonltd
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

5. Start the development server:
```bash
python -m src.main
```

## Project Structure
```
circo-ai/
├── src/
│   ├── vision/          # AI Vision Module
│   ├── robotics/        # Robotic Control Interface
│   ├── maintenance/     # Predictive Maintenance Engine
│   ├── energy/         # Energy Management System
│   ├── blockchain/     # Blockchain Logging Module
│   ├── orchestration/  # Central Orchestration Hub
│   └── dashboard/      # Real-time Analytics Dashboard
├── tests/              # Test suite
├── docs/              # Documentation
├── config/           # Configuration files
└── scripts/          # Utility scripts
```

## Documentation
- [System Architecture](docs/architecture.md)
- [API Documentation](docs/api.md)
- [Deployment Guide](docs/deployment.md)
- [User Manual](docs/user-manual.md)

## Contributing
Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.
## CI/CD Test

   Testing pipeline - [june 3 2025]
## License
This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Contact
For any inquiries, please open an issue in the GitHub repository.

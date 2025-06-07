# Vet Bot

A Python-based veterinary service finder that helps users locate nearby veterinarians based on location and animal type.

## Features

- Find veterinarians by location
- Filter vets by animal specialty
- View vet contact information and available hours
- Image analysis capabilities
- Interactive agent interface

## Setup

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv .venv
   ```
3. Activate the virtual environment:
   - Windows: `.venv\Scripts\activate`
   - Unix/MacOS: `source .venv/bin/activate`
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the main application:
```bash
python main.py
```

## Project Structure

- `main.py` - Main application entry point
- `vetconnect.py` - Core vet finding functionality
- `agent.py` - Interactive agent implementation
- `image_analysis.py` - Image processing capabilities
- `data/` - Contains vet database and other data files 
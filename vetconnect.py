import json
import os

# Load vet directory
def load_vets():
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct path to vets.json
    vets_path = os.path.join(script_dir, "data", "vets.json")
    with open(vets_path) as f:
        return json.load(f)

# Find vets by location and animal type
def find_nearby_vets(location=None, animal_type=None):
    vets = load_vets()
    matches = []

    # Normalize inputs
    location = location.lower() if location else None
    animal_type = animal_type.lower() if animal_type else None

    for vet in vets:
        # Check if vet's location matches (case-insensitive)
        location_match = location and vet["location"].lower() == location
        
        # Check if vet handles this animal type (case-insensitive)
        specialty_match = animal_type and any(s.lower() == animal_type for s in vet["specialty"])

        if location_match and specialty_match:
            # Format available hours if present
            hours_info = f"\nAvailable: {vet['available_hours']}" if "available_hours" in vet else ""
            
            # Add formatted vet info to matches
            matches.append({
                "name": vet["name"],
                "contact": vet["contact"],
                "address": vet["address"] + hours_info
            })

    return matches

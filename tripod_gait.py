import hexapod # Import the updated Hexapod class
import numpy as np

# --- Simulation Parameters ---
XML_FILENAME = "hexapod.xml" # Ensure this file exists
RENDER_SIMULATION = True     # Set to True to watch the simulation
SIMULATION_DURATION = 20.0   # How long to run (in simulation seconds)

def main():
    print(f"Loading model from: {XML_FILENAME}")
    try:
        # Create the Hexapod simulation object
        sim = hexapod.Hexapod(xml_path=XML_FILENAME)
        sim.simend = SIMULATION_DURATION # Set simulation duration

    except FileNotFoundError as e:
        print(e)
        return
    except Exception as e:
        print(f"Error initializing simulation: {e}")
        # If using Linux, you might need to install mesa drivers:
        # sudo apt-get update && sudo apt-get install -y libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf
        return

    print("Resetting simulation...")
    initial_obs = sim.reset()
    print(f"Initial Observation shape: {initial_obs.shape}")
    print("-" * 30)

    print("Starting simulation...")
    sim.simulate(render=RENDER_SIMULATION) # Pass render flag here

    print("Simulation finished.")

if __name__ == "__main__":
    main()
import cv2
from qibullet import SimulationManager
from qibullet import PepperVirtual
from qibullet import Camera


if __name__ == "__main__":
    simulation_manager = SimulationManager()
    client = simulation_manager.launchSimulation(gui=True)
    pepper = simulation_manager.spawnPepper(client, spawn_ground_plane=True)
    
    # Subscribing to the bottom RGB camera, with a QVGA resolution at 15 frames
    # per second. When subscribing, the resolution and the fps are respectively
    # set to QVGA and 30 fps by default 
    handle = pepper.subscribeCamera(
        PepperVirtual.ID_CAMERA_TOP,
        resolution=Camera.K_QVGA,
        fps=30.0)

    # Add other objects to the simulation...

    try:
        while True:
            # Retrieving and displaying the synthetic image using the subscription
            # handle and OpenCV
            img = pepper.getCameraFrame(handle)
            cv2.imshow("synthetic top camera", img)
            cv2.waitKey(1)
    except KeyboardInterrupt:
        pass
    finally:
        pepper.unsubscribeCamera(handle)
        simulation_manager.stopSimulation(handle)
    
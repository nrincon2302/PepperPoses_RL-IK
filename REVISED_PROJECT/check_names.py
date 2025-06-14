import pybullet as p
from qibullet import SimulationManager

def inspect_robot():
    """
    Imprime todos los nombres de joints y links de Pepper para diagnóstico.
    """
    simulation_manager = SimulationManager()
    client = simulation_manager.launchSimulation(gui=False)
    print("Spawneando a Pepper...")
    pepper = simulation_manager.spawnPepper(client, spawn_ground_plane=True)
    robot_id = pepper.getRobotModel()
    print(f"\nPepper cargado con ID: {robot_id}")
    print("-" * 50)
    print(f"{'Index':<7} | {'Joint Name':<25} | {'Link Name':<25}")
    print("-" * 50)

    for i in range(p.getNumJoints(robot_id, physicsClientId=client)):
        info = p.getJointInfo(robot_id, i, physicsClientId=client)
        joint_index = info[0]
        joint_name = info[1].decode('utf-8')
        link_name = info[12].decode('utf-8')
        
        # Imprime solo los que tienen nombres de link relevantes
        if link_name:
             print(f"{joint_index:<7} | {joint_name:<25} | {link_name:<25}")

    simulation_manager.stopSimulation(client)
    print("-" * 50)
    print("Inspección completa.")

if __name__ == "__main__":
    inspect_robot()
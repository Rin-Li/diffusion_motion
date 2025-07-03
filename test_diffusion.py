import numpy as np
from utils.value_utils import create_test_scenario, generate_path, show_multiple_with_collision_colors
from core.diffusion.diffusion import PlaneDiffusionPolicy
from core.networks.embeddUnet import ConditionalUnet1D
from config.plane_test_embeed import PlaneTestEmbedConfig

def test_diffusion_policy(policy, num_tests=20, device='cuda'):
    """
    Test diffusion policy using value_utils functions
    """
    # Test parameters
    bounds = np.array([[0.0, 8.0], [0.0, 8.0]])
    cell_size = 1.0
    origin = bounds[:, 0]
    max_rectangles = (3, 5)
    rng = np.random.default_rng(40)
    
    # Storage for results
    grid_list = []
    path_list = []
    start_list = []
    goal_list = []
    
    print(f"Generating {num_tests} test scenarios...")
    
    for i in range(num_tests):
        try:
            # Create test scenario using your value_utils function
            start, goal, obstacles = create_test_scenario(
                bounds=bounds,
                cell_size=cell_size,
                origin=origin,
                rng=rng,
                max_rectangles=max_rectangles,
                device=device
            )
            
            # Generate path using diffusion policy
            trajectory, _ = generate_path(policy, start, goal, obstacles)
            
            # Convert to numpy for visualization
            grid_np = obstacles[0, 0].cpu().numpy()
            path_np = trajectory[0].cpu().numpy()
            start_np = start[0].cpu().numpy()
            goal_np = goal[0].cpu().numpy()
            
            # Store results
            grid_list.append(grid_np)
            path_list.append(path_np)
            start_list.append(start_np)
            goal_list.append(goal_np)
            
            print(f"Test {i+1}/{num_tests} completed")
            
        except Exception as e:
            print(f"Error in test {i+1}: {str(e)}")
            continue
    
    print(f"Generated {len(path_list)} valid test cases")
    
    # Visualize results with collision-based coloring
    indices = list(range(len(path_list)))
    results = show_multiple_with_collision_colors(
        grid_list, path_list, start_list, goal_list, indices, cols=5
    )
    
    return results


def main():
    """
    Main function to load model and run tests
    """
    # Model configuration
    ckpt_path = '/Users/yulinli/Desktop/Exp/diffusion_policy/ckpt_ep999.ckpt'
    device = "cpu"  # Change to "cuda" if available
    
    print("Loading diffusion policy model...")
    
    # Initialize model
    config = PlaneTestEmbedConfig()
    config_dict = config.to_dict()
    
    net = ConditionalUnet1D(
        input_dim=config.network_config["unet_config"]["action_dim"],
        global_cond_dim=config.network_config["vit_config"]["num_classes"] + 
                        config.network_config["mlp_config"]["embed_dim"],
        network_config=config.network_config
    )
    
    from core.diffusion.diffusion import build_noise_scheduler_from_config
    scheduler = build_noise_scheduler_from_config(config_dict)
    
    policy = PlaneDiffusionPolicy(
        model=net, 
        noise_scheduler=scheduler, 
        config=config_dict, 
        device=device
    )
    
    # Load weights
    policy.load_weights(ckpt_path)
    print("Model loaded successfully!")
    
    # Run tests
    print("\nStarting diffusion policy evaluation...")
    results = test_diffusion_policy(
        policy, num_tests=25, device=device
    )
    
    print("\n=== Final Results ===")
    print(f"Success Rate: {results['success_rate']:.2%}")
    print(f"Collision Rate: {results['collision_rate']:.2%}")
    
    return results


if __name__ == "__main__":
    results = main()
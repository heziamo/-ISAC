from models.isac_sat_env import make_env

env = make_env()()
obs = env.reset()
print("初始观测:", obs)

for i in range(5):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    print(f"Step {i+1}:")
    print("  动作:", action)
    print("  观测:", obs)
    print("  奖励:", reward)
    print("  info:", info)
    if done:
        break

env.render()

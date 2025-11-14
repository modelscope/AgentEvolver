### Step 1. Basic Dependency Installation

Make sure you have **conda** and **cuda toolkit** installed.

Then, set up the training environment by running the script

```bash
bash install.sh
```


### Step 2. Setup Env-Service (Appworld as example)
The script below sets up an environment for appworld.

```bash
cd env_service/environments/appworld && bash setup.sh
```
For other environment setup, refer to [docs/guidelines/env_service.md](../guidelines/env_service.md) 📄

### Step 3. Setup ReMe (Optional)
Set up the ReMe for experience management by running the script:
```bash
bash external/reme/install_reme.sh
```
For more detailed installation, please refer to [ReMe](https://github.com/agentscope-ai/ReMe).
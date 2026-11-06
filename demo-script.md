# Short Demo Script

Estimated time: 3–5 minutes. Use only a local demonstration; do not connect live machinery.

## Before the demo

1. From the project root, activate the environment and start FastAPI:

   ```powershell
   .\venv\Scripts\Activate.ps1
   python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8001
   ```

2. In a second terminal, start the React app:

   ```powershell
   cd .\frontend
   npm.cmd run dev -- --host 127.0.0.1
   ```

3. Open <http://127.0.0.1:5173>. For API documentation, open <http://127.0.0.1:8001/docs>.

If model artifacts are not present, train them first using the commands in the root README.

## Demonstration steps

1. **Overview:** show machine count, saved readings, latest model signals, and the temperature trend. Explain that stored data is local SQLite data.
2. **Submit a reading:** open Sensor readings and use a clearly labelled demo machine ID, for example `demo-press-01`.
3. **Analyze:** enter the following synthetic example and select “Analyze & save reading”:

   ```json
   {
     "product_type": "L",
     "air_temperature_k": 300.0,
     "process_temperature_k": 308.0,
     "rotational_speed_rpm": 1300,
     "torque_nm": 50.0,
     "tool_wear_min": 230
   }
   ```

   Explain that analysis stores the reading and result, and that the displayed classifier output is not calibrated real-world risk.

4. **Investigation:** show rule IDs, observed values, interpretations, suggested checks, and limitations. Emphasize that a matched synthetic rule is a prompt for review, not a diagnosis.
5. **Maintenance review:** record a sample action such as “Reviewed sensor values,” add a reviewer name, and show that the action appears in the audit list.
6. **Close:** restate that AI4I is synthetic, the Autoencoder may miss failures, and the project is not a certified safety system.

## Resetting demo data

Use a unique demo machine ID so it is easy to identify. Do not delete the SQLite database to reset a presentation if it contains other project data; back it up and remove only the demo machine's related rows if cleanup is necessary.

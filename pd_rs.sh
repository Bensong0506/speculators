#! /bin/bash
cd /opt/route-server-mm0510/
HOST_IP="0.0.0.0"
kill -9 $(pgrep -f route)
export PREFILL_SERVERS="${HOST_IP}:8010"
export DECODE_SERVERS="${HOST_IP}:8020"
rm -f /home/ma-user/AscendCloud/log/route-server.log
bash deploy.sh --rs_name route_server_skip --port 19800 --mode singleton \
        --infer-mode Layerwise-MM  --infer-module-version pipeline --log-file-path /home/ma-user/AscendCloud/log/route-server.log --reuse-prefilled-tokens

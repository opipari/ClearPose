for model_sensor in primesense D435 L515; do
for dataset_sensor in primesense D435 L515; do
    if [ "$dataset_sensor" != "primesense" ] || [ "$model_sensor" != "primesense" ]; then
        echo "model_sensor: " $model_sensor ", dataset_sensor: " $dataset_sensor
    fi
done
done

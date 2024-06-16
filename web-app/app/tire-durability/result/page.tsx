'use client'

import { useEffect, useState } from "react";
import {Spinner} from "@nextui-org/spinner";
import {Card, CardHeader, CardFooter} from "@nextui-org/card";
import {Divider} from "@nextui-org/divider";

export default function TireDurabilityResult() {
  const [totalDurability, setTotalDurability] = useState(0);
  const [distinctTraveled, setDistinctTraveled] = useState(0);
  useEffect(() => {
    const fetchDurabilityPrediction = async () => {
      const tireData = sessionStorage.getItem('tireData');
      const tireDataParsed = tireData && JSON.parse(tireData);
    
      const {
        averageSpeedCity,
        averageSpeedOutsideCity,
        carWeight,
        diameter,
        distanceDriven,
        drivingStyle,
        extraLoad,
        highestTemperature,
        lowestTemperature,
        offRoadQuality,
        pavedRoadQuality,
        pressureCheckFrequency,
        ratio,
        ridingOffRoad,
        ridingOnPavedRoad,
        threadDepth,
        type,
        usageFrequency,
        usedYears,
        usualWeatherConditions,
        width,
      } = tireDataParsed

      setDistinctTraveled(distanceDriven)
    
      const response = await fetch(`https://0llke9850j.execute-api.eu-west-2.amazonaws.com/predict?thread_depth=${threadDepth}&tire_type=${type}&tire_width=${width}&tire_diameter=${diameter}&tire_ratio=${ratio}&car_weight=${carWeight}&pressure_checks_frequency=${pressureCheckFrequency}&city_avg_speed=${averageSpeedCity}&outside_city_avg_speed=${averageSpeedOutsideCity}&driving_style=${drivingStyle}&paved_road=${ridingOnPavedRoad}&offroad=${ridingOffRoad}&paved_road_quality=${pavedRoadQuality}&offroad_quality=${offRoadQuality}&min_temperature=${lowestTemperature}&max_temperature=${highestTemperature}&avg_temperature=${usualWeatherConditions}&driving_frequency=${usageFrequency}&car_extra_load_weight=${extraLoad}&tire_age=${usedYears}&distance_driven_with_tires=${distanceDriven}`);
    
      const responseParsed = await response.json();
      
      setTotalDurability(responseParsed.result);
    }
    fetchDurabilityPrediction();

  }, [])
  return (
    <section className="flex flex-col items-center justify-center gap-4 py-8 md:py-10">
      {!totalDurability ? (
        <Spinner />
      ): (
        <Card className="max-w-[400px]">
          <CardHeader className="flex gap-3">
            <p>Total durability (km): {totalDurability}</p>
          </CardHeader>
          <Divider/>
          <CardFooter>
            <p>Remaining distance (km): {totalDurability - distinctTraveled}</p>
          </CardFooter>
        </Card>
      )}
    </section>
  );
}

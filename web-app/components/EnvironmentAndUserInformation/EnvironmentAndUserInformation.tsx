'use client'

import { Slider } from "@nextui-org/slider";
import { Input } from "@nextui-org/input";
import { RadioGroup, Radio } from "@nextui-org/radio";
import { title } from "../primitives";
import { EnvironmentAndUserData } from "@/types";
import { useState } from "react";
import { Button } from "@nextui-org/button";

type EnvironmentAndUserInformationProps = {
  setEnvironmentAndUserInformation: (data: EnvironmentAndUserData) => void;
}

export default function EnvironmentAndUserInformation({ setEnvironmentAndUserInformation }: EnvironmentAndUserInformationProps) {
  const [environmentData, setEnvironmentData] = useState<EnvironmentAndUserData>({
    pressureCheckFrequency: 5,
    averageSpeedCity: 0,
    averageSpeedOutsideCity: 0,
    drivingStyle: 5,
    ridingOnPavedRoad: 5,
    pavedRoadQuality: 5,
    ridingOffRoad: 5,
    offRoadQuality: 5,
    lowestTemperature: 0,
    highestTemperature: 0,
    usualWeatherConditions: 5,
    usageFrequency: 5,
    extraLoad: 5,
    distanceDriven: 0,
  });

  const handleChange = (event: React.ChangeEvent<HTMLInputElement | HTMLInputElement>) => {
    const { name, value } = event.target;
    setEnvironmentData((prevData) => ({
      ...prevData,
      [name]: parseFloat(value),
    }));
  };

  const handleSliderChange = (name: string, value: number | number[]) => {
    setEnvironmentData((prevData) => ({
      ...prevData,
      [name]: typeof value === 'number' ? value : value[0],
    }));
  }

  const handleSave = () => {
    setEnvironmentAndUserInformation(environmentData);
  };

  return (
    <>
      <Slider   
        size="md"
        step={1}
        color="foreground"
        label="Pressure checks frequency"
        showSteps={true} 
        maxValue={10} 
        minValue={0} 
        defaultValue={5}
        className="max-w-md"
        onChange={(v) => handleSliderChange('pressureCheckFrequency', v)}
      />
      <Input onChange={handleChange} type="number" label="Average speed in the city" name="averageSpeedCity" />
      <Input onChange={handleChange} type="number" label="Average speed outside of the city" name="averageSpeedOutsideCity" />
      <Slider   
        size="md"
        step={1}
        color="foreground"
        label="Driving style (0 - Least aggressive, 10 - Most aggressive)"
        showSteps={true} 
        maxValue={10} 
        minValue={0} 
        defaultValue={5}
        className="max-w-md"
        onChange={(v) => handleSliderChange('drivingStyle', v)}
      />
      <Slider   
        size="md"
        step={1}
        color="foreground"
        label="Riding on paved road (0 - Less, 10 - More)"
        showSteps={true} 
        maxValue={10} 
        minValue={0} 
        defaultValue={5}
        className="max-w-md"
        onChange={(v) => handleSliderChange('ridingOnPavedRoad', v)}
      />
      <Slider   
        size="md"
        step={1}
        color="foreground"
        label="Usual paved road quality (0 - Worst, 10 - Best)"
        showSteps={true} 
        maxValue={10} 
        minValue={0} 
        defaultValue={5}
        className="max-w-md" 
        onChange={(v) => handleSliderChange('pavedRoadQuality', v)}
      />
      <Slider   
        size="md"
        step={1}
        color="foreground"
        label="Riding on off-road (0 - Less, 10 - More)"
        showSteps={true} 
        maxValue={10} 
        minValue={0} 
        defaultValue={5}
        className="max-w-md" 
        onChange={(v) => handleSliderChange('ridingOffRoad', v)}
      />
      <Slider   
        size="md"
        step={1}
        color="foreground"
        label="Usual off-road quality (0 - Worst, 10 - Best)"
        showSteps={true} 
        maxValue={10} 
        minValue={0} 
        defaultValue={5}
        className="max-w-md" 
        onChange={(v) => handleSliderChange('offRoadQuality', v)}
      />
      <Input onChange={handleChange} type="number" label="Lowest temperature weather conditions (deg celsius)" name="lowestTemperature" />
      <Input onChange={handleChange} type="number" label="Highest temperature weather conditions (deg celsius)" name="highestTemperature" />
      <Slider   
        size="md"
        step={1}
        color="foreground"
        label="Usual weather conditions (0 - Cold, 10 - Warm)"
        showSteps={true} 
        maxValue={10} 
        minValue={0} 
        defaultValue={5}
        className="max-w-md" 
        onChange={(v) => handleSliderChange('usualWeatherConditions', v)}
      />
      <Slider   
        size="md"
        step={1}
        color="foreground"
        label="Usage frequency (0 - Never, 10 - Often)"
        showSteps={true} 
        maxValue={10} 
        minValue={0} 
        defaultValue={5}
        className="max-w-md" 
        onChange={(v) => handleSliderChange('usageFrequency', v)}
      />
      <Slider   
        size="md"
        step={1}
        color="foreground"
        label="Extra load (0 - Less, 10 - More)"
        showSteps={true} 
        maxValue={10} 
        minValue={0} 
        defaultValue={5}
        className="max-w-md" 
        onChange={(v) => handleSliderChange('extraLoad', v)}
      />
      <Input onChange={handleChange} type="number" label="Distance driven (km)" name="distanceDriven" />
      <Button onClick={handleSave}>Save Changes</Button>
    </>
  );
}

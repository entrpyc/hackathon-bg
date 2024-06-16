import { SVGProps } from "react";

export type IconSvgProps = SVGProps<SVGSVGElement> & {
  size?: number;
};

export type TireData = {
  threadDepth: number;
  width: number;
  diameter: number;
  ratio: number;
  carWeight: number;
  usedYears: number;
  type: 'summer' | 'winter' | 'all-season';
};

export type EnvironmentAndUserData = {
  pressureCheckFrequency: number;
  averageSpeedCity: number;
  averageSpeedOutsideCity: number;
  drivingStyle: number;
  ridingOnPavedRoad: number;
  pavedRoadQuality: number;
  ridingOffRoad: number;
  offRoadQuality: number;
  lowestTemperature: number;
  highestTemperature: number;
  usualWeatherConditions: number;
  usageFrequency: number;
  extraLoad: number;
  distanceDriven: number;
};
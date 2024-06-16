'use client'

import { useState } from 'react';
import { Input } from "@nextui-org/input";
import { RadioGroup, Radio } from "@nextui-org/radio";
import { title } from "../primitives";
import { Button } from '@nextui-org/button';
import { TireData } from '@/types';

type TireInformationProps = {
  setTireInformation: (data: TireData) => void;
}

export default function TireInformation({ setTireInformation }: TireInformationProps) {
  const [tireData, setTireData] = useState<TireData>({
    threadDepth: 0,
    width: 0,
    diameter: 0,
    ratio: 0,
    carWeight: 0,
    usedYears: 0,
    type: 'all-season',
  });

  const handleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = event.target;
    setTireData((prevData) => ({
      ...prevData,
      [name]: value,
    }));
  };

  const handleSelectChange = (name: string, value: string) => {
    setTireData((prevData) => ({
      ...prevData,
      [name]: value,
    }));
  };


  const handleSave = () => {
    setTireInformation(tireData);
  };

  return (
    <>
      <Input onChange={handleChange} type="number" label="Used (years)" name="usedYears" />
      <Input
        type="number"
        label="Thread Depth (mm)"
        name="threadDepth"
        onChange={handleChange}
      />
      <Input
        type="number"
        label="Width (mm)"
        name="width"
        onChange={handleChange}
      />
      <Input
        type="number"
        label="Diameter (mm)"
        name="diameter"
        onChange={handleChange}
      />
      <Input type="number" label="Ratio (mm)" name="ratio" onChange={handleChange} />
      <Input type="number" label="Car weight (kg)" name="carWeight" onChange={handleChange} />
      <RadioGroup label="Type" value={tireData.type} onValueChange={(v) => handleSelectChange('type', v)} name="type">
        <Radio value="all-season">All season</Radio>
        <Radio value="summer">Summer</Radio>
        <Radio value="winter">Winter</Radio>
      </RadioGroup>
      <Button onClick={handleSave}>Save Changes</Button>
    </>
  );
}

'use client'
import { Button } from "@nextui-org/button";
import { Accordion, AccordionItem } from "@nextui-org/accordion";
import EnvironmentAndUserInformation from "@/components/EnvironmentAndUserInformation/EnvironmentAndUserInformation";
import TireInformation from "@/components/TireInformation/TireInformation";
import { SetStateAction, useEffect, useState } from "react";
import { EnvironmentAndUserData, TireData } from "@/types";

export default function Predict() {
  const [tireInformation, setTireInformation] = useState<TireData>();
  const [environmentAndUserInformation, setEnvironmentAndUserInformation] = useState<EnvironmentAndUserData>();
  const [selectedKeys, setSelectedKeys] = useState(new Set(["1"]));


  // double check data types
  // 3. fetch request to endpoint for durability approximation
  // 4. display remaining time and total durability approximation
  // update logo
  // add onboarding

  const handleSubmit = () => {
    const formData = {
      ...tireInformation,
      ...environmentAndUserInformation,
    };
  
    const serializedData = JSON.stringify(formData);
  
    sessionStorage.setItem('tireData', serializedData);
  
    window.location.href = '/tire-durability/result';
  };

  useEffect(() => {
    if(!tireInformation) setSelectedKeys(new Set(["1"]));
    if(tireInformation && !environmentAndUserInformation) setSelectedKeys(new Set(["2"]));
    if(tireInformation && environmentAndUserInformation) setSelectedKeys(new Set(["3"]));
  }, [tireInformation, environmentAndUserInformation])

  return (
    <section className="flex flex-col justify-center gap-20 py-8 md:py-10 w-[500px] mx-auto">
      <Accordion selectionMode="single" variant="splitted" defaultExpandedKeys={["1"]} selectedKeys={selectedKeys} onSelectionChange={(v) => setSelectedKeys(v as SetStateAction<Set<string>>)}>
        <AccordionItem key="1" aria-label="Tire specifications" title="Tire specifications">
          <div className="flex flex-col justify-center gap-4">
            <TireInformation setTireInformation={setTireInformation}  />
          </div>
        </AccordionItem>
        <AccordionItem key="2" aria-label="Environment and User behavior" title="Environment and User behavior">
          <div className="flex flex-col justify-center gap-4">
            <EnvironmentAndUserInformation
              setEnvironmentAndUserInformation={setEnvironmentAndUserInformation}
            />
          </div>
        </AccordionItem>
      </Accordion>
      

      {tireInformation && environmentAndUserInformation && (
        <Button onClick={handleSubmit}>Predict durability with AI</Button>
      )}
    </section>
  );
}

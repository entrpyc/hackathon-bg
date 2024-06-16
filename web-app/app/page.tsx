import { semititle, title } from "@/components/primitives";

export default function Predict() {
  return (
    <section className="flex flex-col gap-44 py-8 md:py-10">
      <div className="flex flex-col gap-3 w-[500px] text-center mx-auto mb-9">
        <h1 className={title()}>Introducing the Tire Wear Prediction System:</h1>
        <p className="w-[500px] mx-auto text-center">Develop models to anticipate tire wear and ensure safety during extended journeys, even with the limitations of the Uniform Tire Quality Grading (UTQG) system.</p>
      </div>


      <div className="flex gap-24 justify-between">
        <img className="opacity-25" src="/slide-2.jpg" alt="" />
        <div className="flex flex-col">
          <h2 className={semititle()}>The problem</h2>

          <p className="mb-20">UTQG offers a standardized, yet approximate, estimate for tire durability. These general tests don&apos;t account for individual driving habits and environmental factors, which significantly impact tire wear.</p>

          <h2 className={semititle()}>The solution</h2>

          <p>Our system utilizes AI to adapt durability assessments to your specific driving habits and environment. We provide a simple and accessible application to easily check your tires&apos; condition at any time.</p>
        </div>
      </div>

      <div className="flex gap-24 justify-between">
        <img src="/slide-1.png" alt="" />
        <div className="flex flex-col">
          <h2 className={semititle()}>Impact</h2>

          <p className="mb-20">Reduce the risk of accidents and ensure safety by predicting potential durability concerns.</p>

          <h2 className={semititle()}>The future of the project</h2>

          <ul className="mb-20 flex flex-col gap-2">
            <li>- Integrate built-in systems to automatically collect tire data.</li>
            <li>- Provide automatic notifications for potential durability concerns.</li>
            <li>- Optimize the market for better tire suggestions based on your specific needs and environment.</li>
          </ul>

          <h2 className={semititle()}>Technologies</h2>

          <p>Utilized and fine-tuned the DistilBert model for its efficiency and effectiveness, providing a highly scalable and accurate solution.</p>
          <p>Set up an environment for quick and easy extension and seamless integration by leveraging on Vercel and Next.js 14.</p>
        </div>
      </div>

      
      
      
      <div className="flex flex-col gap-3 text-center mx-auto mb-9">
        <h2 className={semititle()}>Conclusion</h2>

        <p>Our product is impacting an estimate of 2 billion customers and bringing an innovation to a 232 billion dollar industry.</p>
        <p>Take the opportunity to be a part of this change by trusting your vote to our team :)</p>
      </div>
    </section>
  );
}

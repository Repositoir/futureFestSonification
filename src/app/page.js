"use client"; // Required for Next.js Client Components

import Image from "next/image";
import styles from "./page.module.css";
import { useEffect, useRef } from "react";

export default function Home() {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;

    const particlesArray = [];
    const numberOfParticles = 200;

    // Define the Particle class
    class Particle {
      constructor() {
        this.x = Math.random() * canvas.width;
        this.y = Math.random() * canvas.height;
        this.size = Math.random() * 5 + 1;
      }

      draw() {
        ctx.fillStyle = "rgba(255, 255, 255, 0.7)";
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
        ctx.closePath();
        ctx.fill();
      }
    }

    // Initialize particles
    function initParticles() {
      for (let i = 0; i < numberOfParticles; i++) {
        const particle = new Particle();
        particlesArray.push(particle);
        particle.draw(); // Draw each particle only once
      }
    }

    initParticles(); // Draw all particles once

    // Handle window resize
    window.addEventListener("resize", () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
      ctx.clearRect(0, 0, canvas.width, canvas.height); // Clear canvas on resize
      particlesArray.length = 0; // Clear particle array
      initParticles(); // Redraw particles after resizing
    });
  }, []);

  return (
    <div className={styles.container}>
      <canvas ref={canvasRef} className={styles.canvas}></canvas>
      <main className={styles.main}>
        <div className={styles.titleWrapper}>
          <h1 className={styles.title}>Sonification</h1>
          <h2 className={styles.subtitle}>Image to Sound</h2>
        </div>
      </main>
    </div>
  );
}

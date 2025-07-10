const fs = require('fs');
const ImageTracer = require('./imagetracer_v1.2.6');
const Jimp = require('jimp');
const path = require('path');

(async () => {
    const mlKmeans = require('ml-kmeans');

    if (process.argv.length < 4) {
        console.error("Użycie: node vectorize.js input_image output_svg [numcolors]");
        process.exit(1);
    }

    const inputFile = process.argv[2];
    const outputFile = process.argv[3];
    const numColors = parseInt(process.argv[4]) || 8;

    console.log("📥 Plik wejściowy:", inputFile);
    console.log("📤 Plik wyjściowy:", outputFile);
    console.log("🎨 Liczba kolorów:", numColors);

    if (!fs.existsSync(inputFile)) {
        console.error("❌ Plik nie istnieje.");
        process.exit(1);
    }

    function isCloseColor(c1, c2, tolerance = 30) {
        return Math.abs(c1.r - c2.r) <= tolerance &&
               Math.abs(c1.g - c2.g) <= tolerance &&
               Math.abs(c1.b - c2.b) <= tolerance;
    }

    try {
        const image = await Jimp.read(inputFile);

        // Opcjonalne skalowanie, jeśli potrzebne:
        // await image.scale(2);

        // Usuń tło na podstawie koloru z lewego górnego narożnika
        const bgColorRGBA = Jimp.intToRGBA(image.getPixelColor(0, 0));
        const bgColor = { r: bgColorRGBA.r, g: bgColorRGBA.g, b: bgColorRGBA.b };
        const tolerance = 40;

        image.scan(0, 0, image.bitmap.width, image.bitmap.height, function (x, y, idx) {
            const r = this.bitmap.data[idx + 0];
            const g = this.bitmap.data[idx + 1];
            const b = this.bitmap.data[idx + 2];
            if (isCloseColor({ r, g, b }, bgColor, tolerance)) {
                this.bitmap.data[idx + 3] = 0; // alfa = 0 (przezroczystość)
            }
        });

        // Zbierz piksele do klasteryzacji (losowe próbkowanie)
        const sampledPixels = [];
        const sampleRate = 0.1; // 1% pikseli, zmień w razie potrzeby

        image.scan(0, 0, image.bitmap.width, image.bitmap.height, function (x, y, idx) {
            if (Math.random() > sampleRate) return;

            const a = this.bitmap.data[idx + 3];
            if (a > 50) {
                sampledPixels.push([
                    this.bitmap.data[idx + 0],
                    this.bitmap.data[idx + 1],
                    this.bitmap.data[idx + 2]
                ]);
            }
        });

        console.log("📊 Liczba próbkowanych pikseli:", sampledPixels.length);
        if (sampledPixels.length === 0) {
            console.error("❌ Brak wystarczających pikseli do klasteryzacji!");
            process.exit(2);
        }

        const result = mlKmeans(sampledPixels, numColors);

        if (!result.centroids || !Array.isArray(result.centroids) || result.centroids.length === 0) {
            console.error("❌ Błąd: brak centroidów po klasteryzacji.");
            process.exit(2);
        }

        const centroids = result.centroids
            .filter(c => Array.isArray(c) && c.length === 3);

        if (centroids.length === 0) {
            console.error("❌ Centroidy mają nieprawidłowy format.");
            process.exit(2);
        }

        console.log("🎯 Centroidy:", centroids);

        function nearestCentroidColor(rgb) {
            if (!rgb || rgb.length !== 3) {
                console.warn("⚠️ Nieprawidłowe dane RGB:", rgb);
                return null;
            }

            let minDist = Infinity;
            let minColor = null;

            for (const c of centroids) {
                if (!c || c.length !== 3) continue;

                const dist = Math.sqrt(
                    (rgb[0] - c[0]) ** 2 +
                    (rgb[1] - c[1]) ** 2 +
                    (rgb[2] - c[2]) ** 2
                );

                if (dist < minDist) {
                    minDist = dist;
                    minColor = c;
                }
            }

            return minColor;
        }

        // Zamień kolory na najbliższe centroidy
        image.scan(0, 0, image.bitmap.width, image.bitmap.height, function (x, y, idx) {
            const alpha = this.bitmap.data[idx + 3];
            if (alpha > 50) {
                const oldColor = [
                    this.bitmap.data[idx + 0],
                    this.bitmap.data[idx + 1],
                    this.bitmap.data[idx + 2]
                ];

                const newColor = nearestCentroidColor(oldColor);
                if (!newColor) {
                    console.warn(`⚠️ Nie znaleziono koloru centroidu dla piksela (${x}, ${y})`);
                    return;
                }

                this.bitmap.data[idx + 0] = Math.round(newColor[0]);
                this.bitmap.data[idx + 1] = Math.round(newColor[1]);
                this.bitmap.data[idx + 2] = Math.round(newColor[2]);
            }
        });

        // Przygotuj dane dla ImageTracer
        const { bitmap } = image;
        const imgData = {
            width: bitmap.width,
            height: bitmap.height,
            data: bitmap.data
        };

        const svgstr = ImageTracer.imagedataToSVG(imgData, {
        numberofcolors: numColors,
        ltres: 0.01,      // niższa wartość = większa dokładność linii
        qtres: 0.01,      // niższa wartość = większa dokładność krzywych
        pathomit: 0.2,    // niższa wartość = mniej uproszczeń
        roundcoords: 2,   // więcej miejsc po przecinku
        blurradius: 0,
        blurdelta: 0,
        desc: false
        });


        fs.writeFileSync(outputFile, svgstr);
        console.log("✅ Wektoryzacja zakończona sukcesem:", outputFile);

    } catch (err) {
        console.error("❌ Błąd podczas przetwarzania:", err);
        process.exit(2);
    }
})();

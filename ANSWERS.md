# Answers
Here I will provide some useful notes we may need to know.

## Segmentor
The job of the segmentor as we may infeer is to obtain the mask of the object. But it does so pixel by pixel `(alpha mask)`, so the result is extremely sharp, isolating almost perfectly the body of the person from the background, this is huge when the person has complex clothes or the background is too noisy, etc.

If we do not use a segmentor we can provide the binary mask obtained through YOLO and it will work just fine, the only downside is that we lose that high precision, but we gain resources efficiency, specially if we're running on CPU, since `sam3` seems to be heavy.

**Beneficios de utilizar SAM 3 como segmentor:**

- Ofrece una segmentación ultraprecisa basada en prompts y capacidades de video/imagen avanzada, reduciendo al mínimo los errores de contorno en ropa compleja o fondos confusos.

- Alinea de manera óptima las extremidades con los vértices de la malla 3D, logrando una reconstrucción más limpia.

NOTE: Seems that if we don't have GPU the segmentor is not enabled, investigate further!


## Detector
It's job is to basically detect the person and draw a bounding box.

ViTDet: Es el detector clásico basado en Detectron2, rápido y liviano, pero limitado a detecciones estándar de bounding boxes rectangulares.

SAM 3 (como detector): Al ser un modelo fundacional más moderno y potente, detecta personas con mayor precisión en escenarios difíciles (como oclusiones parciales, posturas extrañas o iluminación compleja).

Cuál es mejor: SAM 3 es ampliamente superior en calidad y robustez, pero ViTDet es más eficiente en recursos. En tu caso particular (corriendo en un Mac con CPU), usar SAM 3 tanto para detectar como para segmentar consumirá muchísima más memoria RAM y tiempo de procesamiento comparado con usar ViTDet.
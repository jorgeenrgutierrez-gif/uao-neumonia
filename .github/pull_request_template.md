## Qué cambia
<!-- Resumen en 1-3 líneas. Enlazar la tarjeta del tablero si aplica. -->

## Módulo(s) afectado(s)
- [ ] load_model  - [ ] read_img  - [ ] preprocess_img
- [ ] grad_cam    - [ ] integrator - [ ] ui  - [ ] infraestructura/docs

## Cómo probarlo
```bash
make install && make check
```

## Checklist del autor
- [ ] `make check` en verde localmente
- [ ] Commits convencionales (`feat:`, `fix:`, `test:`, `docs:`, `refactor:`, `chore:`)
- [ ] `ui.py` sigue importando solo `integrator`
- [ ] Sin `pip`, `requirements.txt`, `.ipynb`, modelos ni datos de pacientes en el diff
- [ ] Pruebas nuevas/actualizadas con patrón AAA

## Revisión (la completa el revisor)
- [ ] Leí el código y entiendo la responsabilidad de cada cambio
- [ ] Ejecuté `make check` en mi máquina
- Comentarios:

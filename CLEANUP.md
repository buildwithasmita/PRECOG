# Pre-Release Cleanup Checklist

## Code Quality
- [ ] Remove all print() statements (use logging instead)
- [ ] Remove commented-out code
- [ ] Remove TODO comments
- [ ] Check all imports (remove unused)
- [ ] Add docstrings to all classes/methods
- [ ] Add type hints to all functions
- [ ] Run black formatter: `black app/`
- [ ] Run isort: `isort app/`

## Configuration
- [ ] No hardcoded paths (use config/settings.py)
- [ ] No API keys in code (use .env)
- [ ] Update .env.example with all variables
- [ ] Verify .gitignore includes:
  * __pycache__/
  * *.pyc
  * .env
  * data/models/*.joblib
  * venv/

## Testing
- [ ] Run all unit tests: `pytest tests/ -v`
- [ ] Test API endpoints manually
- [ ] Test dashboard loads without errors
- [ ] Test Docker build: `docker-compose build`
- [ ] Test Docker run: `docker-compose up`

## Documentation
- [ ] README.md complete with:
  * Problem statement ?
  * Screenshots (add real ones!)
  * Setup instructions ?
  * API examples ?
- [ ] API docs auto-generated (FastAPI /docs)
- [ ] Dashboard spec complete ?
- [ ] Add LICENSE file (MIT)
- [ ] Add CONTRIBUTING.md

## Files to Remove Before Push
- [ ] notebooks/ (or add to .gitignore)
- [ ] Large CSV files in data/raw/ (keep only sample)
- [ ] Test outputs
- [ ] .DS_Store (Mac)

## Performance Optimization
- [ ] Add caching to expensive operations
- [ ] Optimize feature engineering (vectorize operations)
- [ ] Add batch processing for large datasets
- [ ] Add progress bars for long operations

## Security
- [ ] No credentials in code
- [ ] Validate all API inputs
- [ ] Add rate limiting to API
- [ ] Add CORS properly configured

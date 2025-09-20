# Auto-Instrumentation Quick Reference

## 🚀 Essential Commands

### Basic Setup
```python
import brokle.auto_instrumentation as brokle_ai

# Auto-instrument all available libraries
brokle_ai.auto_instrument()

# Check status
brokle_ai.print_status()
```

### Selective Control
```python
# Instrument specific libraries only
brokle_ai.auto_instrument(libraries=["openai", "anthropic"])

# Exclude specific libraries
brokle_ai.auto_instrument(exclude=["langchain"])

# Manual control
brokle_ai.instrument("openai")      # Enable
brokle_ai.uninstrument("openai")    # Disable
```

### Health Monitoring
```python
# Visual status
brokle_ai.print_status()           # Basic status
brokle_ai.print_health_report()    # Detailed health

# Programmatic access
status = brokle_ai.get_status()
health = brokle_ai.get_health_report()
```

### Error Recovery
```python
# Reset all errors and circuit breakers
brokle_ai.reset_all_errors()

# Get error handler for advanced control
error_handler = brokle_ai.get_error_handler()
```

## 📊 Status Indicators

| Symbol | Meaning | Description |
|--------|---------|-------------|
| ✅ | Instrumented | Library is actively instrumented |
| ⚪ | Available | Library installed but not instrumented |
| ❌ | Not Available | Library not installed |
| 🔴 | Failed/Unhealthy | Instrumentation issues detected |
| 💥 | Error | Critical instrumentation error |

## 🏥 Health Scores

- **100%**: Perfect health, all systems operational
- **80-99%**: Good health, minor issues possible
- **50-79%**: Degraded performance, check logs
- **<50%**: Critical issues, immediate attention needed

## ⚡ Circuit Breaker States

| State | Symbol | Meaning |
|-------|--------|---------|
| `closed` | 🟢 | Normal operation |
| `open` | 🔴 | Protection mode (failures detected) |
| `half-open` | 🟡 | Testing recovery |

## 🔧 Troubleshooting Checklist

### Library Not Found
1. Check installation: `pip list | grep openai`
2. Install if missing: `pip install openai`
3. Restart Python session

### API Key Issues
1. Set environment variable: `export BROKLE_API_KEY="your-key"`
2. Verify in code: `print(os.getenv('BROKLE_API_KEY'))`
3. Check configuration: `from brokle import get_client; print(get_client().config.api_key)`

### Circuit Breaker Open
1. Check health: `brokle_ai.print_health_report()`
2. Reset errors: `brokle_ai.reset_all_errors()`
3. Wait for automatic recovery (30 seconds)

### Performance Issues
1. Check error rates in health report
2. Verify network connectivity
3. Consider temporary disable: `brokle_ai.uninstrument_all()`

## 📝 Environment Variables

```bash
# Essential
export BROKLE_API_KEY="your-api-key"
export BROKLE_ORGANIZATION_ID="org_xxx"
export BROKLE_PROJECT_ID="proj_xxx"

# Optional
export BROKLE_BASE_URL="https://api.brokle.ai"
export BROKLE_ENVIRONMENT="production"
export BROKLE_AUTO_INSTRUMENT="true"
```

## 🧪 Testing Commands

```python
# Health check script
def quick_health_check():
    health = brokle_ai.get_health_report()
    score = health["overall_health"]["score"]
    print(f"Health: {score}% {'🟢' if score > 80 else '🟡' if score > 50 else '🔴'}")

    if score < 100:
        brokle_ai.print_health_report()

# Run health check
quick_health_check()
```

## ⚠️ Common Pitfalls

1. **Don't instrument after library import**: Import auto-instrumentation first
   ```python
   # ❌ Wrong
   import openai
   import brokle.auto_instrumentation as brokle_ai
   brokle_ai.auto_instrument()

   # ✅ Correct
   import brokle.auto_instrumentation as brokle_ai
   brokle_ai.auto_instrument()
   import openai
   ```

2. **Don't ignore health status**: Monitor health in production
3. **Don't disable circuit breakers**: They protect your application
4. **Don't panic on errors**: System is designed for graceful degradation

## 🎯 Performance Expectations

- **Overhead**: <1ms per instrumented call
- **Memory**: ~2MB base + 100KB per library
- **Circuit breaker**: Opens after 3 failures, recovers in 30s
- **Health checks**: Complete in <0.1ms

## 💡 Pro Tips

1. **Initialize early**: Set up auto-instrumentation at app startup
2. **Monitor health**: Include health endpoints in your monitoring
3. **Use error recovery**: Implement automatic error recovery in production
4. **Debug with logging**: Enable debug logging for troubleshooting
5. **Test thoroughly**: Always test instrumentation in staging first

---

*Keep this reference handy for quick troubleshooting and setup!*

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
========================================================================
SCRIPT DE PRUEBA PARA LA API
Test del endpoint /predict de la API FastAPI
========================================================================
"""

import requests
import json

# URL de la API
API_URL = "http://127.0.0.1:8000"

def test_health():
    """Prueba el endpoint de salud"""
    print("\n" + "="*70)
    print("TEST 1: Health Check")
    print("="*70)
    
    response = requests.get(f"{API_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    return response.status_code == 200

def test_tipos_procedimiento():
    """Prueba el endpoint de tipos de procedimiento"""
    print("\n" + "="*70)
    print("TEST 2: Tipos de Procedimiento")
    print("="*70)
    
    response = requests.get(f"{API_URL}/tipos-procedimiento")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    
    return response.status_code == 200

def test_predict_caso1():
    """Prueba predicción - Caso 1: Dentro de DRL"""
    print("\n" + "="*70)
    print("TEST 3: Predicción - Caso DENTRO de DRL")
    print("="*70)
    
    payload = {
        "Tipo_Procedimiento": "Coronariografía Diagnóstica",
        "PKA_Gycm2": 45.0,
        "Kar_mGy": 350.0,
        "Tiempo_Fluoroscopia_min": 8.0,
        "Edad": 55,
        "Peso": 70.0
    }
    
    print(f"Payload: {json.dumps(payload, indent=2, ensure_ascii=False)}")
    
    response = requests.post(f"{API_URL}/predict", json=payload)
    print(f"\nStatus Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    
    return response.status_code == 200

def test_predict_caso2():
    """Prueba predicción - Caso 2: Excede DRL"""
    print("\n" + "="*70)
    print("TEST 4: Predicción - Caso EXCEDE DRL")
    print("="*70)
    
    payload = {
        "Tipo_Procedimiento": "Angiografía Cerebral",
        "PKA_Gycm2": 180.0,
        "Kar_mGy": 1200.0,
        "Tiempo_Fluoroscopia_min": 25.0,
        "Edad": 70,
        "Peso": 80.0
    }
    
    print(f"Payload: {json.dumps(payload, indent=2, ensure_ascii=False)}")
    
    response = requests.post(f"{API_URL}/predict", json=payload)
    print(f"\nStatus Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    
    return response.status_code == 200

def test_predict_caso3():
    """Prueba predicción - Caso 3: Angiografía Renal"""
    print("\n" + "="*70)
    print("TEST 5: Predicción - Angiografía Renal")
    print("="*70)
    
    payload = {
        "Tipo_Procedimiento": "Angiografía Renal",
        "PKA_Gycm2": 60.0,
        "Kar_mGy": 450.0,
        "Tiempo_Fluoroscopia_min": 10.0,
        "Edad": 62,
        "Peso": 75.0
    }
    
    print(f"Payload: {json.dumps(payload, indent=2, ensure_ascii=False)}")
    
    response = requests.post(f"{API_URL}/predict", json=payload)
    print(f"\nStatus Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    
    return response.status_code == 200

def run_all_tests():
    """Ejecuta todos los tests"""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*20 + "TEST SUITE - API DRL" + " "*28 + "║")
    print("╚" + "="*68 + "╝")
    
    tests = [
        ("Health Check", test_health),
        ("Tipos de Procedimiento", test_tipos_procedimiento),
        ("Predicción - Dentro DRL", test_predict_caso1),
        ("Predicción - Excede DRL", test_predict_caso2),
        ("Predicción - Angiografía Renal", test_predict_caso3)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except requests.exceptions.ConnectionError:
            print(f"\n❌ ERROR: No se puede conectar con la API en {API_URL}")
            print("Asegúrate de que el servidor esté corriendo:")
            print("  cd api && python main.py")
            return
        except Exception as e:
            print(f"\n❌ ERROR en {test_name}: {str(e)}")
            results.append((test_name, False))
    
    # Resumen
    print("\n" + "="*70)
    print("RESUMEN DE TESTS")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print("="*70)
    print(f"Total: {passed}/{total} tests pasados ({passed/total*100:.1f}%)")
    print("="*70)

if __name__ == "__main__":
    run_all_tests()

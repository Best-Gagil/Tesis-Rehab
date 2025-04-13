package com.ejemplo.bluetooth

import android.Manifest
import android.annotation.SuppressLint
import android.app.Activity
import android.bluetooth.*
import android.bluetooth.le.*
import android.content.Context
import android.content.pm.PackageManager
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.webkit.JavascriptInterface
import androidx.core.app.ActivityCompat
import android.os.ParcelUuid
import androidx.core.content.ContextCompat
import android.bluetooth.BluetoothGatt
import android.bluetooth.BluetoothGattCallback
import android.bluetooth.BluetoothGattCharacteristic
import android.bluetooth.BluetoothGattDescriptor
import android.bluetooth.BluetoothProfile
import android.os.Build
import androidx.annotation.RequiresApi
import java.util.*

@RequiresApi(Build.VERSION_CODES.JELLY_BEAN_MR2)
class BluetoothService(private val activity: Activity) {

    companion object {
        private var instance: BluetoothService? = null

        @JvmStatic
        fun getInstance(activity: Activity): BluetoothService {
            if (instance == null) {
                instance = BluetoothService(activity)
            }
            return instance!!
        }
    }

    private val REQUEST_CODE = 1
    private var stopScanRunnable: Runnable? = null

    private lateinit var ServiceEnviarHex_UUID: UUID
    private lateinit var CharacteristicEnviando_UUID: UUID
    private lateinit var CharacteristicRecibiendo_UUID: UUID
    private lateinit var DescriptorRecibiendo_UUID: UUID
    private lateinit var AdvertisingService_UUID: UUID
    private lateinit var ScanResponse_UUID: UUID

    private var bluetoothGatt: BluetoothGatt? = null
    private val bluetoothAdapter: BluetoothAdapter? by lazy {
        val bluetoothManager = activity.getSystemService(Context.BLUETOOTH_SERVICE) as BluetoothManager
        bluetoothManager.adapter
    }

    private val handler = Handler(Looper.getMainLooper())
    private var scanCallback: ScanCallback? = null


/****************************************************************************************************
*                               FUNCIÓN PRINCIPAL DE ESCANEO Y CONEXIÓN                             *
****************************************************************************************************/
    @RequiresApi(Build.VERSION_CODES.LOLLIPOP)
    @JavascriptInterface
    @SuppressLint("MissingPermission")
    fun scanAndConnect(bleParams: Array<String>) {

        /*if (!hasBluetoothPermissions()) {
            Log.e("AndroidStudio", "[BLE] Faltan permisos de Bluetooth")
            UnitySendMessage("BluetoothManager", "OnError", "Faltan permisos")
            ActivityCompat.requestPermissions(activity,
                arrayOf(Manifest.permission.BLUETOOTH_SCAN, Manifest.permission.BLUETOOTH_CONNECT), REQUEST_CODE)
            return
        }*/

        // Verificar que todos los valores estén llenos
        if (bleParams.size < 7) {
            // Log.e("AndroidStudio", "[BLE] Parámetros incompletos en scanAndConnect")
            UnitySendMessage("BluetoothManager", "OnError", "Parámetros incompletos en scanAndConnect")
            return
        }

        // Asignar los valores recibidos desde Unity a las variables
        val targetMacAddress = bleParams[0]
        ServiceEnviarHex_UUID = UUID.fromString(bleParams[1])
        CharacteristicEnviando_UUID = UUID.fromString(bleParams[2])
        CharacteristicRecibiendo_UUID = UUID.fromString(bleParams[3])
        DescriptorRecibiendo_UUID = UUID.fromString(bleParams[4])
        AdvertisingService_UUID = UUID.fromString(bleParams[5])
        ScanResponse_UUID = UUID.fromString(bleParams[6])

        val bluetoothManager = activity.getSystemService(Context.BLUETOOTH_SERVICE) as BluetoothManager
        val bluetoothAdapter = bluetoothManager.adapter
        val bluetoothLeScanner = bluetoothAdapter.bluetoothLeScanner

        // Verificar que el modulo Bluetooth no esté apagado
        if (bluetoothAdapter == null || !bluetoothAdapter.isEnabled) {
            // Log.e("AndroidStudio", "[BLE] Bluetooth está apagado. No se puede escanear.")
            UnitySendMessage("BluetoothManager", "OnError", "Bluetooth está apagado. No se puede escanear.")
            return
        }

        /*// Verificar que se tenga acceso al escáner BLE
        if (bluetoothAdapter?.bluetoothLeScanner == null) {
            Log.e("AndroidStudio", "[BLE] El escáner BLE no está disponible")
        }*/
        if (bluetoothLeScanner == null) {
            UnitySendMessage("BluetoothManager", "OnError", "No se pudo obtener el escáner BLE")
            return
        }

        val scannedDevices = mutableSetOf<String>()

        /****************************************************************************************************
        *                              CALLBACK DE FUNCIÓN PRINCIPAL                                        *
        ****************************************************************************************************/
        val scanCallback = object : ScanCallback() {
            override fun onScanResult(callbackType: Int, result: ScanResult?) {
                result?.device?.let { device ->

                    // Obtener el Service Data del advertising
                    val scanRecord = result.scanRecord
                    val serviceData = scanRecord?.getServiceData(ParcelUuid(AdvertisingService_UUID))
                    val deviceName = device.name ?: "Desconocido"
                    // Enviar a Unity que este dispositivo es válido
                    UnitySendMessage("BluetoothManager", "OnDeviceFound", "$deviceName|${device.address}|01")
                    Log.d("AndroidStudio", "[BLE] MAC esperada: $targetMacAddress")

                    if (device.address.equals(device.address, ignoreCase = true)) {
                        Log.d("AndroidStudio", "[BLE] Conectando a: ${device.address}")
                        scannedDevices.add(device.address) // Registrar que el dispositivo fue detectado
                        bluetoothAdapter?.bluetoothLeScanner?.stopScan(this)

                        // DETENER ESPERA DE 10 SEGUNDOS DE ESCANEO----------------------
                        handler.removeCallbacks(stopScanRunnable!!)

                        // Obtener la actividad de Unity de forma segura
                        val unityActivity: Activity? = try {
                            val clazz = Class.forName("com.unity3d.player.UnityPlayer")
                            val field = clazz.getDeclaredField("currentActivity")
                            field.isAccessible = true
                            field.get(null) as? Activity
                        } catch (e: Exception) {
                            null
                        }

                        // Usar la actividad obtenida para la conexión BLE
                        bluetoothGatt = device.connectGatt(unityActivity, false, gattCallback)

                        UnitySendMessage("BluetoothManager", "OnConnectionResult", "true")
                    }
                }
            }

            override fun onScanFailed(errorCode: Int) {
                // Log.e("AndroidStudio", "[BLE] Error en el escaneo: $errorCode")
                UnitySendMessage("BluetoothManager", "OnError", "Error en el escaneo: $errorCode")
            }
        }

        // Configuración de ScanSettings
        val scanSettings = ScanSettings.Builder()
            .setScanMode(ScanSettings.SCAN_MODE_LOW_LATENCY)
            .build()

        // Filtro para buscar el UUID 0x2345
        val filter = ScanFilter.Builder()
            .setServiceUuid(ParcelUuid(AdvertisingService_UUID)) // Asegura filtrar dispositivos con ese UUID
            .build()

        val scanFilters = listOf(filter)

        Log.d("AndroidStudio", "[BLE] Iniciando escaneo de dispositivos BLE...") // Confirmación de que el escaneo inicia

        /****************************************************************************************************
        *                               LLAMADO DE CALLBACK DE FUNCIÓN PRINCIPAL                            *
        ****************************************************************************************************/
        bluetoothLeScanner.startScan(scanFilters, scanSettings, scanCallback)

        ///////////////////---------------------------------------------------
        stopScanRunnable = Runnable {
            bluetoothLeScanner?.stopScan(scanCallback)
            if (!scannedDevices.contains(targetMacAddress)) {
                UnitySendMessage("BluetoothManager", "OnScanFinished", "Dispositivo no encontrado: $targetMacAddress")
            }
        }
        handler.postDelayed(stopScanRunnable!!, 10000)
        ///////////////////---------------------------------------------------
        /*
        // Detener escaneo después de 10 segundos en caso de no encontrar
        handler.postDelayed({
            bluetoothLeScanner?.stopScan(scanCallback)
            if (!scannedDevices.contains(targetMacAddress)) {
                UnitySendMessage("BluetoothManager", "OnScanFinished", "Dispositivo no encontrado: $targetMacAddress")
            }
        }, 10000)*/
    }

/****************************************************************************************************
*                               FUNCIÓN PRINCIPAL DE ENVÍO DE DATO HEX                              *
****************************************************************************************************/
    @JavascriptInterface
    fun sendData(hexData: String) {
        /*if (!hasBluetoothPermissions()) {
            Log.e("AndroidStudio", "[BLE] Faltan permisos de Bluetooth")
            UnitySendMessage("BluetoothManager", "OnError", "Faltan permisos")
            return
        }*/

        val gatt = bluetoothGatt
        if (gatt == null) {
            // Log.e("AndroidStudio", "[BLE] No hay conexión GATT activa")
            UnitySendMessage("BluetoothManager", "OnError", "No hay conexión GATT activa")
            return
        }

        val service = gatt.getService(ServiceEnviarHex_UUID)
        val characteristic = service?.getCharacteristic(CharacteristicEnviando_UUID)

        if (characteristic == null) {
            // Log.e("AndroidStudio", "[BLE] Característica de escritura no encontrada")
            UnitySendMessage("BluetoothManager", "OnError", "Característica de escritura no encontrada")
            return
        }

        characteristic.value = hexStringToByteArray(hexData)

        if (ContextCompat.checkSelfPermission(activity, Manifest.permission.BLUETOOTH_CONNECT)
            == PackageManager.PERMISSION_GRANTED) {
            val result = gatt.writeCharacteristic(characteristic)
            if (!result) {
                // Log.e("AndroidStudio", "[BLE] Error al escribir en la característica")
                UnitySendMessage("BluetoothManager", "OnError", "Error al escribir en la característica")
            } else {
                // Log.d("AndroidStudio", "[BLE] Datos enviados correctamente: $hexData")
                UnitySendMessage("BluetoothManager", "OnDataSent", hexData)
            }
        } else {
            // Log.e("AndroidStudio", "[BLE] Faltan permisos de Bluetooth para escribir en la característica")
            UnitySendMessage("BluetoothManager", "OnError", "Faltan permisos de Bluetooth para escribir en la característica")
        }
    }

/****************************************************************************************************
*                               FUNCIÓN PRINCIPAL DE DESCONEXIÓN DEL DISPOSITIVO                    *
****************************************************************************************************/
    @JavascriptInterface
    fun disconnect() {
        if (ActivityCompat.checkSelfPermission(activity, Manifest.permission.BLUETOOTH_CONNECT) == PackageManager.PERMISSION_GRANTED) {
            bluetoothGatt?.disconnect()
            bluetoothGatt?.close()
            UnitySendMessage("BluetoothManager", "OnDeviceDisconnected", "Disconnected")

        } else {
            // Log.e("AndroidStudio", "[BLE] No tienes permisos para usar Bluetooth")
            UnitySendMessage("BluetoothManager", "OnError", "No tienes permisos para usar Bluetooth")
            ActivityCompat.requestPermissions(activity, arrayOf(Manifest.permission.BLUETOOTH_CONNECT), REQUEST_CODE)
        }
    }

    /****************************************************************************************************
    *                  FUNCIÓN SECUNDARIA DE CONVERSIÓN DE DATO HEX COMO BYTE ARRAY                     *
    ****************************************************************************************************/
    private fun hexStringToByteArray(s: String): ByteArray {
        return try {
            s.chunked(2).map { it.toInt(16).toByte() }.toByteArray()
        } catch (e: NumberFormatException) {
            Log.e("AndroidStudio", "[BLE] Error al convertir hexadecimal a byte array: ${e.message}")
            byteArrayOf()
        }
    }

    // BORRRRRAAAAAAARLLAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA?????????????????????
    private fun hasBluetoothPermissions(): Boolean {
        return ContextCompat.checkSelfPermission(activity, Manifest.permission.BLUETOOTH_SCAN) == PackageManager.PERMISSION_GRANTED &&
                    ContextCompat.checkSelfPermission(activity, Manifest.permission.BLUETOOTH_CONNECT) == PackageManager.PERMISSION_GRANTED
    }

    /****************************************************************************************************
    *                  FUNCIÓN SECUNDARIA DE ENVÍO DE MENSAJES A UNITY                                  *
    ****************************************************************************************************/
    private fun UnitySendMessage(gameObject: String, method: String, message: String) {
        try {
            // Log.d("AndroidStudio", "[BLE] Enviando mensaje a Unity: $gameObject - $method - $message")
            val unityPlayer = Class.forName("com.unity3d.player.UnityPlayer")
            val unityInstance = unityPlayer.getField("currentActivity")[null]
            val unitySendMessage = unityPlayer.getMethod("UnitySendMessage", String::class.java, String::class.java, String::class.java)
            unitySendMessage.invoke(null, gameObject, method, message)
        } catch (e: Exception) {
            Log.e("AndroidStudio", "[BLE] Error enviando mensaje a Unity: ${e.message}", e)
        }
    }

    /****************************************************************************************************
     *                  FUNCIÓN SECUNDARIA DE ENVÍO DE MENSAJES A UNITY                                  *
     ****************************************************************************************************/
    private val gattCallback = object : BluetoothGattCallback() {
        override fun onConnectionStateChange(gatt: BluetoothGatt?, status: Int, newState: Int) {
            if (newState == BluetoothProfile.STATE_CONNECTED) {
                Log.d("AndroidStudio", "[BLE] Conexión GATT exitosa con ${gatt?.device?.address}, descubriendo servicios...")

                if (ActivityCompat.checkSelfPermission(activity, Manifest.permission.BLUETOOTH_CONNECT) == PackageManager.PERMISSION_GRANTED) {
                    val success = gatt?.discoverServices()
                    // Log.d("AndroidStudio", "[BLE] Intento de descubrimiento de servicios iniciado: $success")
                } else {
                    Log.e("AndroidStudio", "[BLE] Permiso BLUETOOTH_CONNECT no concedido")
                }

            } else if (newState == BluetoothProfile.STATE_DISCONNECTED) {
                Log.e("AndroidStudio", "[BLE] Desconectado de GATT con ${gatt?.device?.address}, cerrando conexión...")
                // UnitySendMessage("BluetoothManager", "OnDeviceDisconnected", "Disconnected")
                bluetoothGatt?.close()
                bluetoothGatt = null
            } else {
                Log.e("AndroidStudio", "[BLE] Cambio de estado desconocido en GATT: $newState con código de estado: $status")
            }
        }

        override fun onServicesDiscovered(gatt: BluetoothGatt?, status: Int) {
            if (status == BluetoothGatt.GATT_SUCCESS) {

                gatt?.let {
                    enableNotifications(it) // 👈 Habilitar notificaciones después de descubrir servicios
                }

                Log.d("AndroidStudio", "[BLE] Servicios descubiertos con éxito en ${gatt?.device?.address}")

                val service = gatt?.getService(ServiceEnviarHex_UUID)
                if (service != null) {
                    Log.d("AndroidStudio", "[BLE] Servicio encontrado: ${ServiceEnviarHex_UUID}")

                    val characteristic = service.getCharacteristic(CharacteristicRecibiendo_UUID)
                    if (characteristic != null) {
                        Log.d("AndroidStudio", "[BLE] Característica de recepción encontrada: ${CharacteristicRecibiendo_UUID}")

                        if (ActivityCompat.checkSelfPermission(activity, Manifest.permission.BLUETOOTH_CONNECT) == PackageManager.PERMISSION_GRANTED) {
                            gatt.setCharacteristicNotification(characteristic, true)
                            // Log.d("AndroidStudio", "[BLE] Notificaciones activadas en la característica")

                            val descriptor = characteristic.getDescriptor(DescriptorRecibiendo_UUID)
                            if (descriptor != null) {
                                descriptor.value = BluetoothGattDescriptor.ENABLE_NOTIFICATION_VALUE
                                gatt.writeDescriptor(descriptor)
                                // Log.d("AndroidStudio", "[BLE] Descriptor de notificaciones escrito correctamente")
                            } else {
                                Log.e("AndroidStudio", "[BLE] Descriptor no encontrado en la característica")
                            }
                        } else {
                            Log.e("AndroidStudio", "[BLE] Permiso BLUETOOTH_CONNECT no concedido para activar notificaciones")
                        }
                    } else {
                        Log.e("AndroidStudio", "[BLE] Característica de recepción NO encontrada en el servicio")
                    }
                } else {
                    Log.e("AndroidStudio", "[BLE] Servicio NO encontrado: ${ServiceEnviarHex_UUID}")
                }

            } else {
                Log.e("AndroidStudio", "[BLE] Error al descubrir servicios: $status")
            }
        }

        override fun onCharacteristicChanged(gatt: BluetoothGatt?, characteristic: BluetoothGattCharacteristic?) {

            Log.d("AndroidStudio", "[BLE] onCharacteristicChanged llamado") // 👈 PRIMER LOG

            if(characteristic?.uuid == CharacteristicRecibiendo_UUID){
                characteristic?.value?.let { data ->
                    //-------------------------val value = characteristic.value // ← aquí está el mensaje en bytes
                    val hexString = data.joinToString("") { String.format("%02X", it) }

                    // Log.d("AndroidStudio", "[BLE] Datos recibidos: $hexString")
                    UnitySendMessage("BluetoothManager", "OnDataReceived", hexString)
                }
            } else {
                Log.d("AndroidStudio", "[BLE] Notificación ignorada, UUID no coincide: ${characteristic?.uuid}")
            }
        }
    }

    /****************************************************************************************************
    *                  FUNCIÓN PRINCIPAL DE ACCESO A NOTIFICACIONES PARA DISPOSITIVO BLE                *
    ****************************************************************************************************/
    fun enableNotifications(gatt: BluetoothGatt) {

        if (ActivityCompat.checkSelfPermission(activity, Manifest.permission.BLUETOOTH_CONNECT) != PackageManager.PERMISSION_GRANTED) {
            Log.e("AndroidStudio", "Permiso BLUETOOTH_CONNECT no concedido")
            return
        }

        val service = gatt.getService(ServiceEnviarHex_UUID) // 👈 UUID del servicio ya definido
        val characteristic = service?.getCharacteristic(CharacteristicRecibiendo_UUID) // 👈 UUID de la característica para recibir

        if (characteristic != null) {
            gatt.setCharacteristicNotification(characteristic, true)

            val descriptor = characteristic.getDescriptor(DescriptorRecibiendo_UUID) // 👈 UUID del descriptor ya definido
            descriptor.value = BluetoothGattDescriptor.ENABLE_NOTIFICATION_VALUE
            gatt.writeDescriptor(descriptor)

            Log.d("AndroidStudio", "[BLE] Notificaciones habilitadas en UUID: ${characteristic.uuid}")
        } else {
            Log.e("AndroidStudio", "[BLE] Característica con UUID $CharacteristicRecibiendo_UUID no encontrada")
        }
    }


}
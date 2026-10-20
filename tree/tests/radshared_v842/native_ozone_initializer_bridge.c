/* Framework routing only. MPAS mpas_atmphys_init.F:397 calls the complete
 * mpas_atmphys_o3climatology.F:59-194 initializer with two pool references.
 * GNU --wrap preserves the pristine scientific objects and the default call.
 * The native adapter serializes installation/call/removal under its RLock. */
typedef int (*ozone_initializer)(void *, void *);
static ozone_initializer selected;
static int callback_status;
static int callback_calls;

extern void __real___mpas_atmphys_o3climatology_MOD_init_o3climatology(void *, void *);

void native_ozone_initializer_set(ozone_initializer callback) {
    selected = callback;
    callback_status = 0;
    callback_calls = 0;
}

int native_ozone_initializer_status(void) { return callback_status; }
int native_ozone_initializer_calls(void) { return callback_calls; }

void __wrap___mpas_atmphys_o3climatology_MOD_init_o3climatology(void *mesh, void *atm_input) {
    if (selected) {
        ++callback_calls;
        callback_status = selected(mesh, atm_input);
    } else {
        __real___mpas_atmphys_o3climatology_MOD_init_o3climatology(mesh, atm_input);
    }
}
